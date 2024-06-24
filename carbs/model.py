import math
from typing import List
from typing import Optional

import attr
import numpy as np
import pyro
import torch
from pyro.contrib import gp as gp
from pyro.contrib.gp.kernels import Kernel
from pyro.contrib.gp.models import GPRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from torch import Tensor
from torch.distributions import Normal

from carbs.utils import ObservationInBasic
from carbs.utils import OutstandingSuggestionEstimatorEnum
from carbs.utils import SuggestionInBasic
from carbs.utils import SurrogateModelParams


@attr.s(auto_attribs=True, collect_by_mro=True)
class SurrogateObservationOutputs:
    surrogate_output: Tensor
    surrogate_var: Tensor
    cost_estimate: Tensor
    target_estimate: Tensor
    target_var: Tensor
    success_probability: Tensor
    pareto_surrogate: Optional[Tensor]
    pareto_estimate: Optional[Tensor]


class SurrogateModel:
    def __init__(
        self,
        params: SurrogateModelParams,
    ) -> None:
        self.params = params
        self.output_transformer: Optional[QuantileTransformer] = None
        self.cost_transformer: Optional[MinMaxScaler] = None
        self.output_model: Optional[GPRegression] = None
        self.cost_model: Optional[GPRegression] = None
        self.pareto_model: Optional[GPRegression] = None
        self.min_logcost: float = float("-inf")
        self.max_logcost: float = float("inf")
        self.min_pareto_logcost: float = float("-inf")
        self.max_pareto_logcost: float = float("inf")
        self.success_model: Optional[GPRegression] = None

    def _get_kernel(self) -> Kernel:
        matern_kernel = gp.kernels.Matern32(
            input_dim=self.params.real_dims,
            lengthscale=self.params.scale_length * torch.ones((self.params.real_dims,)),
        )
        linear_kernel = gp.kernels.Linear(input_dim=self.params.real_dims)
        return gp.kernels.Sum(linear_kernel, matern_kernel)

    def _get_model(self, inputs: Tensor, outputs: Tensor, kernel: Optional[Kernel] = None) -> GPRegression:
        # Heavily inspired by the HEBO paper. Some choices that appear arbitrary, such as the model noise, are copied from that paper.
        # https://arxiv.org/abs/2012.03826
        if kernel is None:
            assert inputs.shape[-1] == self.params.real_dims, f"Must provide kernel for input shape {inputs.shape}"
            kernel = self._get_kernel()
        # Gaussian Process Regression
        model = gp.models.GPRegression(
            inputs.to(self.params.device),
            outputs.to(self.params.device),
            kernel=kernel,
            jitter=1.0e-4,
        ).to(self.params.device)
        model.noise = pyro.nn.PyroSample(pyro.distributions.LogNormal(math.log(1e-2), 0.5))

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        gp.util.train(model, optimizer)
        model.eval()
        return model

    def fit_observations(self, success_observations: List[ObservationInBasic]) -> None:
        self._fit_target_transformers(success_observations)

        inputs_in_basic = torch.stack([x.real_number_input for x in success_observations]).detach()
        outputs_in_surrogate = self._target_to_surrogate(torch.tensor([x.output for x in success_observations]))
        self.output_model = self._get_model(inputs_in_basic, outputs_in_surrogate)

        costs_after_transform = self._cost_to_logcost(torch.tensor([x.cost for x in success_observations]))
        self.cost_model = self._get_model(inputs_in_basic, costs_after_transform)

    def _fit_target_transformers(self, success_observations: List[ObservationInBasic]) -> None:
        raw_outputs = np.array([x.output for x in success_observations])
        # Using n_quantiles < len(observations_in_basic) preserves more distance information within the quantiles
        n_quantiles = int(np.sqrt(len(success_observations)))
        self.output_transformer = QuantileTransformer(output_distribution="normal", n_quantiles=n_quantiles)
        self.output_transformer.fit(raw_outputs.reshape(-1, 1))

        log_costs = np.log(np.array([x.cost for x in success_observations]))
        self.cost_transformer = MinMaxScaler(feature_range=(-1, 1))
        self.cost_transformer.fit(log_costs.reshape(-1, 1))
        transformed_costs = self.cost_transformer.transform(log_costs.reshape(-1, 1))
        self.min_logcost = transformed_costs.min()
        self.max_logcost = transformed_costs.max()

    # Target space is the space of the score function that we're optimizing.
    # Surrogate space is a well-behaved Gaussian space. It's the output analog of basic space.s
    def _target_to_surrogate(self, x: Tensor) -> Tensor:
        # Goes from target function space to the surrogate function space we model with GP
        assert self.output_transformer is not None, "Fit output_transformer before calling target_to_surrogate!"
        x_shape = x.shape
        x = x.view(-1, 1).cpu()
        transformed_x = torch.from_numpy(
            self.output_transformer.transform(x.numpy()) * self.params.better_direction_sign
        ).to(self.params.device)
        return transformed_x.view(*x_shape)

    def _surrogate_to_target(self, x: Tensor) -> Tensor:
        # TODO: a good unit test would be that this inverts the above
        assert self.output_transformer is not None, "Fit output_transformer before calling surrogate_to_target!"
        x_shape = x.shape
        x = x.view(-1, 1).cpu()
        transformed_x = torch.from_numpy(
            self.output_transformer.inverse_transform(x.numpy() * self.params.better_direction_sign)
        ).to(self.params.device)
        return transformed_x.view(*x_shape)

    # Cost space is in seconds or dollars or something like that.
    # Logcost space is the log of that, scaled a min-max range of -1 to 1.
    def _cost_to_logcost(self, x: Tensor) -> Tensor:
        # Goes from cost space to the logcost function space we model with GP
        assert self.cost_transformer is not None, "Fit cost_transformer before calling cost_to_logcost!"
        x_shape = x.shape
        x = torch.log(x.view(-1, 1)).cpu()
        transformed_x = torch.from_numpy(self.cost_transformer.transform(x.numpy())).to(self.params.device)
        return transformed_x.view(*x_shape)

    def _logcost_to_cost(self, x: Tensor) -> Tensor:
        # TODO: a good unit test would be that this inverts the above
        assert self.cost_transformer is not None, "Fit cost_transformer before calling logcost_to_cost!"
        x_shape = x.shape
        transformed_x = torch.from_numpy(self.cost_transformer.inverse_transform(x.view(-1, 1).cpu().numpy())).to(
            self.params.device
        )
        return torch.exp(transformed_x).view(*x_shape)

    def fit_suggestions(self, outstanding_suggestions: List[SuggestionInBasic]) -> None:
        # Note that although we are fitting, no target outputs are passed in, and they have to be estimated using the surrogate model.
        if len(outstanding_suggestions) == 0:
            return

        assert self.output_model is not None, "Fit observations before suggestions!"
        assert self.cost_model is not None, "Fit observations before suggestions!"

        inputs_in_basic = (
            torch.stack([x.real_number_input for x in outstanding_suggestions]).detach().to(self.params.device)
        )

        # In both mean and Thompson sampling, the output model, which was trained on all real observations, is used to estimate the outputs of the suggestions. This is necessary to do as we await the real outputs.
        if self.params.outstanding_suggestion_estimator == OutstandingSuggestionEstimatorEnum.MEAN:
            output_predictions, _unused_output_vars = self.output_model(
                inputs_in_basic, full_cov=False, noiseless=True
            )
        elif self.params.outstanding_suggestion_estimator == OutstandingSuggestionEstimatorEnum.THOMPSON:
            sampler = self.output_model.iter_sample(noiseless=True)
            outputs = []
            for input_in_basic in inputs_in_basic:
                outputs.append(sampler(input_in_basic.unsqueeze(0)))
            output_predictions = torch.cat(outputs, dim=0)
        else:
            raise Exception(f"Invalid estimator {self.params.outstanding_suggestion_estimator}")
        # With these new guesses at the outputs, we retrain the output model, which tells the Bayesian optimizer not to look for uncertainty near these points.
        combined_inputs = torch.cat([self.output_model.X, inputs_in_basic])
        combined_outputs = torch.cat([self.output_model.y, output_predictions])
        self.output_model = self._get_model(combined_inputs, combined_outputs)

    def fit_pareto_set(self, pareto_observations: List[ObservationInBasic]) -> None:
        costs_after_transform = self._cost_to_logcost(torch.tensor([x.cost for x in pareto_observations]))
        outputs_in_surrogate = self._target_to_surrogate(torch.tensor([x.output for x in pareto_observations]))
        # TODO(research): This model is not guaranteed to be monotonic, which would be nice
        self.pareto_model = self._get_model(
            costs_after_transform, outputs_in_surrogate, kernel=gp.kernels.RBF(input_dim=1)
        )
        self.min_pareto_logcost = costs_after_transform.min().item()
        self.max_pareto_logcost = costs_after_transform.max().item()

    def get_pareto_surrogate_for_cost(self, cost: float) -> float:
        assert self.pareto_model is not None
        costs_after_transform = self._cost_to_logcost(torch.tensor([cost]))
        costs_after_transform = torch.clamp(
            costs_after_transform, min=self.min_pareto_logcost, max=self.max_pareto_logcost
        )
        pareto_output, pareto_var = self.pareto_model(costs_after_transform)
        return float(pareto_output.item())

    def fit_failures(
        self, success_observations: List[ObservationInBasic], failure_observations: List[ObservationInBasic]
    ) -> None:
        num_success, num_failure = len(success_observations), len(failure_observations)
        if num_failure == 0 or num_success == 0:
            self.success_predictor = None
            return

        all_observations = success_observations + failure_observations
        inputs_in_basic = torch.stack(([x.real_number_input for x in all_observations])).detach()
        # Use this labeling so we can just get the CDF at zero as probability of success
        labels = torch.tensor([-1] * num_success + [1] * num_failure)

        self.success_model = self._get_model(inputs_in_basic, labels)

    def _get_success_prob(self, samples_in_natural: Tensor):
        if self.success_model is None:
            return torch.ones((samples_in_natural.shape[0],), device=self.params.device)

        success_pred, success_var = self.success_model(samples_in_natural)
        prior = Normal(success_pred, success_var)
        success_pred = prior.cdf(torch.zeros((samples_in_natural.shape[0],), device=self.params.device))
        return success_pred

    @torch.no_grad()
    def observe_surrogate(self, samples_in_basic: Tensor) -> SurrogateObservationOutputs:
        assert self.output_model is not None
        surrogate_output, surrogate_var = self.output_model(samples_in_basic.to(self.params.device))
        assert self.cost_model is not None
        logcost, logcost_var = self.cost_model(samples_in_basic.to(self.params.device))
        if self.pareto_model is not None:
            pareto_logcost = torch.clamp(logcost, min=self.min_pareto_logcost, max=self.max_pareto_logcost)
            pareto_surrogate, pareto_var = self.pareto_model(pareto_logcost)
            pareto_estimate = self._surrogate_to_target(pareto_surrogate)
        else:
            pareto_surrogate = None
            pareto_estimate = None

        success_probability = self._get_success_prob(samples_in_basic.to(self.params.device))

        cost_estimate = self._logcost_to_cost(logcost)
        target_estimate = self._surrogate_to_target(surrogate_output)
        # target_var is only used for logging, but we want it to be correct-ish
        # self._surrogate_to_target will get clamped at the top or bottom, so we take half the distance between
        # +1 and -1 std dev as the new std dev. Return variance which is std dev ** 2
        target_var = torch.square(
            (
                self._surrogate_to_target(surrogate_output + torch.sqrt(surrogate_var))
                - self._surrogate_to_target(surrogate_output - torch.sqrt(surrogate_var))
            )
            / 2.0
        )
        return SurrogateObservationOutputs(
            surrogate_output=surrogate_output,
            surrogate_var=surrogate_var,
            cost_estimate=cost_estimate,
            pareto_surrogate=pareto_surrogate,
            pareto_estimate=pareto_estimate,
            target_estimate=target_estimate,
            target_var=target_var,
            success_probability=success_probability,
        )
