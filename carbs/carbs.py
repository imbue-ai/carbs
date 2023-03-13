# %%
import base64
import math
import random
import traceback
import uuid
from collections import OrderedDict
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import cast

import numpy as np
import torch
import wandb
from attr import evolve
from carbs.model import SurrogateModel
from carbs.model import SurrogateObservationOutputs
from carbs.utils import SUGGESTION_ID_DICT_KEY
from carbs.utils import CARBSParams
from carbs.utils import CategoricalTuple
from carbs.utils import ObservationInBasic
from carbs.utils import ObservationInParam
from carbs.utils import ObserveOutput
from carbs.utils import ParamDictType
from carbs.utils import ParamSpace
from carbs.utils import RealNumberSpace
from carbs.utils import SuggestOutput
from carbs.utils import SuggestionInBasic
from carbs.utils import SurrogateModelParams
from carbs.utils import add_dict_key_prefix
from carbs.utils import aggregate_logical_and_across_dim
from carbs.utils import assert_empty
from carbs.utils import expected_improvement
from carbs.utils import get_pareto_curve_plot
from carbs.utils import get_pareto_groups
from carbs.utils import get_pareto_groups_conservative
from carbs.utils import group_observations
from carbs.utils import load_latest_checkpoint_from_wandb_run
from carbs.utils import observation_group_cost
from carbs.utils import observation_group_output
from carbs.utils import ordered_dict_index
from carbs.utils import pareto_area_from_groups
from loguru import logger
from torch import Tensor
from torch.distributions import Categorical
from torch.distributions import Distribution
from torch.distributions import Normal


class CARBS:
    """
    C.A.R.B.S = Cost Aware (pareto-) Regional Bayesian Search

    Definitions
        Target function: the real function we're trying to find optimal hyperparameters for
        Surrogate (fitness): the estimated output of the target function from the Gaussian process model
        Param space: original space of all the hyperparameters before transforms to the given target function
        Basic space: intermediate space that contains the actual input space of our target function
            e.g. Learning rate in param space is any real positive number and in basic space we reduce the search range
              to a log space `lr = LogSpace(max=1)`
    """

    def __init__(self, params: CARBSParams, param_space_by_name: dict[str, ParamSpace]) -> None:
        logger.info(f"Running CARBS with params {params}")
        self.params = params
        self.param_space_by_name = param_space_by_name
        self._real_number_space_by_name = OrderedDict(
            (k, dim) for k, dim in param_space_by_name.items() if isinstance(dim, RealNumberSpace)
        )
        assert len(self._real_number_space_by_name) == len(param_space_by_name), "Real numbers only supported now"
        self.real_dim = len(self._real_number_space_by_name)
        self.search_center_in_basic = torch.zeros((self.real_dim,))
        self.search_radius_in_basic = torch.tensor(float(self.params.initial_search_radius))

        self.observations_in_basic: List[ObservationInBasic] = []
        self.failed_observations_in_basic: List[ObservationInBasic] = []

        self.min_bounds_in_basic = torch.tensor(
            [dim.basic_from_param(dim.min_bound) for dim in self._real_number_space_by_name.values()]
        )
        self.max_bounds_in_basic = torch.tensor(
            [dim.basic_from_param(dim.max_bound) for dim in self._real_number_space_by_name.values()]
        )
        self.outstanding_suggestions: Dict[str, SuggestionInBasic] = {}

        self._set_seed(params.seed)

        self.resample_history: List[SuggestionInBasic] = []

        # TODO maybe should be able to set this in config
        self.overgenerate_candidate_factor = 2 ** (self.real_dim // 2)

        self.wandb_run = None
        self._init_wandb()

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = f"cuda:{torch.cuda.current_device()}"

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def set_search_center(self, input_in_param: ParamDictType) -> None:
        self.search_center_in_basic = self._param_space_real_to_basic_space_real(input_in_param)

    def _get_mask_for_invalid_points_in_basic(self, input_in_basic: Tensor) -> Tensor:
        is_above_min_bounds = aggregate_logical_and_across_dim(input_in_basic > self.min_bounds_in_basic.unsqueeze(0))
        is_below_max_bounds = aggregate_logical_and_across_dim(input_in_basic < self.max_bounds_in_basic.unsqueeze(0))
        mask = torch.logical_and(is_above_min_bounds, is_below_max_bounds)
        return mask

    def _round_integer_values_in_basic(self, input_in_basic: Tensor) -> Tensor:
        for idx, space in enumerate(self._real_number_space_by_name.values()):
            input_in_basic[..., idx] = space.round_tensor_in_basic(input_in_basic[..., idx])
        return input_in_basic

    def _param_space_real_to_basic_space_real(self, input_in_param: ParamDictType) -> Tensor:
        return torch.tensor(
            [dim.basic_from_param(input_in_param[k]) for k, dim in self._real_number_space_by_name.items()]
        )

    def _param_space_to_basic_space(self, observation_in_params: ObservationInParam) -> ObservationInBasic:
        input_in_param = observation_in_params.input
        suggestion_id = input_in_param.pop(SUGGESTION_ID_DICT_KEY, None)
        real_number_input = self._param_space_real_to_basic_space_real(input_in_param)
        return ObservationInBasic(
            real_number_input=real_number_input,
            output=float(observation_in_params.output),
            cost=float(observation_in_params.cost),
            suggestion_id=suggestion_id,
        )

    def _basic_space_to_param_space(self, real_number_input: Tensor) -> ParamDictType:
        suggestion_in_param: ParamDictType = {}
        suggestion_in_param.update(
            {
                k: dim.param_from_basic(float(v))
                for (k, dim), v in zip(self._real_number_space_by_name.items(), real_number_input)
            }
        )
        return suggestion_in_param

    def _basic_space_to_unrounded_param_space(self, real_number_input: Tensor) -> ParamDictType:
        suggestion_in_param: ParamDictType = {}
        suggestion_in_param.update(
            {
                k: dim.param_from_basic(float(v), is_rounded=False)
                for (k, dim), v in zip(self._real_number_space_by_name.items(), real_number_input)
            }
        )
        return suggestion_in_param

    def _basic_space_to_param_space_for_suggestion(
        self, suggestion_in_basic: SuggestionInBasic, is_remembered: bool, suggestion_id: Optional[str]
    ) -> ParamDictType:
        suggestion_in_param = self._basic_space_to_param_space(suggestion_in_basic.real_number_input)
        if is_remembered:
            if suggestion_id is None:
                suggestion_id = str(uuid.uuid4())
            suggestion_in_param[SUGGESTION_ID_DICT_KEY] = suggestion_id
            self.outstanding_suggestions[suggestion_id] = suggestion_in_basic

        return suggestion_in_param

    def _add_observation(self, observation_in_param: ObservationInParam) -> Optional[ObservationInBasic]:
        observation_in_basic = self._param_space_to_basic_space(observation_in_param)
        if (
            observation_in_param.is_failure
            or not np.isfinite(observation_in_basic.output)
            or np.isnan(observation_in_basic.output)
        ):
            self.failed_observations_in_basic.append(observation_in_basic)
            return None

        self.observations_in_basic.append(observation_in_basic)
        return observation_in_basic

    @property
    def _search_distribution_in_basic(self) -> Distribution:
        return Normal(0, self.search_radius_in_basic)

    def _sample_around_origins_in_basic(self, num_samples: int, origins_in_basic: Tensor) -> Tensor:
        assert (
            len(origins_in_basic.shape) == 2 and origins_in_basic.shape[1] == self.real_dim
        ), f"Bad shape for origins_in_natural: {origins_in_basic.shape}"
        origin_index = Categorical(logits=torch.zeros((origins_in_basic.shape[0],))).sample((num_samples,))
        origin_samples = origins_in_basic[origin_index]
        real_samples: Tensor = origin_samples + self._search_distribution_in_basic.sample((num_samples, self.real_dim))
        return real_samples

    def _get_probability_in_search_space(
        self,
        input_in_basic: Tensor,
        origins_in_basic: Tensor,
    ) -> Tensor:
        input_distance = torch.norm(input_in_basic.unsqueeze(1) - origins_in_basic.unsqueeze(0), dim=-1).to(
            self.device
        )
        real_relative_probability_per_origin = torch.exp(self._search_distribution_in_basic.log_prob(input_distance))
        # we take the max probability origin
        real_relative_probability, _unused_max_index = torch.max(real_relative_probability_per_origin, dim=-1)
        return real_relative_probability

    def initialize_from_observations(self, observations: List[ObservationInParam]) -> ObserveOutput:
        # TODO: where is this used, do we want to keep it?
        logs: Dict[str, Any] = {}
        for observation in observations:
            self._add_observation(observation)
            logs.update(self._get_observation_log(observation))

        if len(observations) > 0:
            best_obs = max(observations, key=lambda x: x.output * self.params.better_direction_sign)
            self.set_search_center(best_obs.input)
        else:
            logger.warning("No way to set the search center!")

        return ObserveOutput(logs=logs)

    def observe_suggestion(self, suggestion_id: str, result: float):
        input_in_param = self._basic_space_to_param_space_for_suggestion(
            self.outstanding_suggestions[suggestion_id], is_remembered=False, suggestion_id=None
        )
        observation = ObservationInParam(input=input_in_param, output=result)
        return self.observe(observation)

    @logger.catch()
    def observe(self, new_observation_in_param: ObservationInParam) -> ObserveOutput:
        """
        Observe a new data point

        Argument:
        new_observation_in_param: ObservationInParam(
            input: ParamDictType -- Dictionary mapping search vars to the values used for input to the function
            output: float -- Target function output
            cost: float = 1.0 -- Usually the time in seconds that the target function took to run
            is_failure: bool = False -- Should be True if training did not complete properly (eg OOM error)
        )
        """
        self.forget_suggestion(new_observation_in_param.input)
        new_observation_in_basic = self._add_observation(new_observation_in_param)
        logs = self._get_observation_log(new_observation_in_param)

        if self.params.is_saved_on_every_observation:
            self._autosave()

        return ObserveOutput(logs=logs)

    def sample_search_space(self, num_samples: int) -> List[SuggestionInBasic]:
        if len(self.observations_in_basic) > self.params.num_random_samples:
            pareto_groups = self._get_pareto_groups(is_conservative=self.params.is_pareto_group_selection_conservative)
            origins_in_basic = torch.stack([x[0].real_number_input for x in pareto_groups], dim=0)
        else:
            origins_in_basic = self.search_center_in_basic.unsqueeze(0)
        samples_in_basic = self._sample_around_origins_in_basic(
            num_samples * self.overgenerate_candidate_factor, origins_in_basic
        )

        valid_sample_mask = self._get_mask_for_invalid_points_in_basic(samples_in_basic)
        samples_in_basic = samples_in_basic[valid_sample_mask][:num_samples]

        if samples_in_basic.shape[0] < num_samples:
            logger.warning(
                f"Undergenerated valid samples, requested {num_samples} generated {samples_in_basic.shape[0]}"
            )

        suggestions_in_basic = [SuggestionInBasic(real_number_input=x) for x in samples_in_basic]

        return suggestions_in_basic

    @torch.no_grad()
    def evaluate_candidates(
        self, samples_in_basic: Tensor, sampled_category: CategoricalTuple, is_rounding_candidates: bool = False
    ) -> Tuple[SurrogateObservationOutputs, Tensor, Tensor, Tensor]:
        surrogate_model = self.get_surrogate_model()
        surrogate_model.fit_observations(self.observations_in_basic)
        surrogate_model.fit_suggestions(list(self.outstanding_suggestions.values()))
        surrogate_model.fit_failures(self.observations_in_basic, self.failed_observations_in_basic)
        pareto_groups = self._get_pareto_groups(is_conservative=self.params.is_pareto_group_selection_conservative)
        all_pareto_observations = [obs for group in pareto_groups for obs in group]
        surrogate_model.fit_pareto_set(all_pareto_observations)

        origins_in_basic = torch.stack([x[0].real_number_input for x in pareto_groups], dim=0)
        #  do rounding first
        if is_rounding_candidates:
            samples_in_basic = self._round_integer_values_in_basic(samples_in_basic)

        probabilities = self._get_probability_in_search_space(
            input_in_basic=samples_in_basic,
            origins_in_basic=origins_in_basic,
        )
        surrogate_model_outputs = surrogate_model.observe_surrogate(samples_in_basic)
        assert surrogate_model_outputs.pareto_surrogate is not None
        ei_value = expected_improvement(
            surrogate_model_outputs.surrogate_output,
            surrogate_model_outputs.surrogate_var,
            best_mu=surrogate_model_outputs.pareto_surrogate,
            exploration_bias=self.params.exploration_bias,
        )
        acquisition_function_value = (
            ei_value
            * probabilities
            / surrogate_model_outputs.cost_estimate
            * surrogate_model_outputs.success_probability
        )

        return (surrogate_model_outputs, probabilities, ei_value, acquisition_function_value)

    @torch.no_grad()
    def _generate_candidate(self) -> Optional[SuggestionInBasic]:
        surrogate_model = self.get_surrogate_model()
        surrogate_model.fit_observations(self.observations_in_basic)
        surrogate_model.fit_suggestions(list(self.outstanding_suggestions.values()))
        surrogate_model.fit_failures(self.observations_in_basic, self.failed_observations_in_basic)
        pareto_groups = self._get_pareto_groups(is_conservative=self.params.is_pareto_group_selection_conservative)
        all_pareto_observations = [obs for group in pareto_groups for obs in group]
        surrogate_model.fit_pareto_set(all_pareto_observations)

        num_samples_to_generate = self.params.num_candidates_for_suggestion_per_dim * self.real_dim

        origins_in_basic = torch.stack([x[0].real_number_input for x in pareto_groups], dim=0)
        samples_in_basic = self._sample_around_origins_in_basic(
            num_samples_to_generate * self.overgenerate_candidate_factor, origins_in_basic
        )

        #  do rounding first
        samples_in_basic = self._round_integer_values_in_basic(samples_in_basic)

        if samples_in_basic.shape[0] == 0:
            return None

        valid_sample_mask = self._get_mask_for_invalid_points_in_basic(samples_in_basic)
        samples_in_basic = samples_in_basic[valid_sample_mask][:num_samples_to_generate]

        if samples_in_basic.shape[0] < num_samples_to_generate:
            logger.info(
                f"Undergenerated valid samples, requested {num_samples_to_generate} generated {samples_in_basic.shape[0]}"
            )
            # sometimes there are just no good candidates left. Stop overgenerating at a factor of 2**real_dim
            self.overgenerate_candidate_factor = min(self.overgenerate_candidate_factor * 2, 2**self.real_dim)

        probabilities = self._get_probability_in_search_space(
            input_in_basic=samples_in_basic, origins_in_basic=origins_in_basic
        )
        surrogate_model_outputs = surrogate_model.observe_surrogate(samples_in_basic)
        assert surrogate_model_outputs.pareto_surrogate is not None
        assert surrogate_model_outputs.pareto_estimate is not None

        pareto_surrogate = surrogate_model_outputs.pareto_surrogate
        if self.params.is_expected_improvement_pareto_value_clamped:
            # Clamping technique to encourage higher cost exploration:
            # Choose a random cost along pareto front, and make the pareto value there the minimum for all observations
            log_min_cost = math.log(observation_group_cost(pareto_groups[0]))
            if len(pareto_groups) > 1:
                log_max_cost = math.log(observation_group_cost(pareto_groups[-1]))
            else:
                log_max_cost = log_min_cost

            # TODO: Naming has become truly convoluted here
            if self.params.max_cost is not None:
                log_max_cost = min(log_max_cost, math.log(self.params.max_cost))

            cost_threshold = math.exp(random.uniform(log_min_cost, log_max_cost))

            pareto_surrogate_at_threshold = surrogate_model.get_pareto_surrogate_for_cost(cost_threshold)
            pareto_surrogate = torch.clamp(pareto_surrogate, min=pareto_surrogate_at_threshold)

        if self.params.is_expected_improvement_value_always_max:
            # NB: Only used for ablations
            best_pareto_group_output = observation_group_output(pareto_groups[-1])
            best_output_in_surrogate = surrogate_model._target_to_surrogate(
                torch.tensor([best_pareto_group_output])
            ).item()
            pareto_surrogate = torch.ones_like(pareto_surrogate) * best_output_in_surrogate

        max_cost_masking = torch.ones_like(surrogate_model_outputs.cost_estimate)
        if self.params.max_cost is not None:
            max_cost_masking = torch.where(
                surrogate_model_outputs.cost_estimate > self.params.max_cost,
                torch.zeros_like(surrogate_model_outputs.cost_estimate),
                torch.ones_like(surrogate_model_outputs.cost_estimate),
            )
            assert max_cost_masking.max().item() > 0.5, f"No candidates below max cost bound {self.params.max_cost}"

        ei_value = expected_improvement(
            surrogate_model_outputs.surrogate_output,
            surrogate_model_outputs.surrogate_var,
            best_mu=pareto_surrogate,
            exploration_bias=self.params.exploration_bias,
        )
        acquisition_function_value = (
            ei_value * probabilities * surrogate_model_outputs.success_probability * max_cost_masking
        )

        best_idx = int(torch.argmax(acquisition_function_value).item())
        log_info = dict(
            surrogate_output=surrogate_model_outputs.surrogate_output[best_idx].item(),
            pareto_surrogate=surrogate_model_outputs.pareto_surrogate[best_idx].item(),
            surrogate_var=surrogate_model_outputs.surrogate_var[best_idx].item(),
            pareto_estimate=surrogate_model_outputs.pareto_estimate[best_idx].item(),
            cost_estimate=surrogate_model_outputs.cost_estimate[best_idx].item(),
            target_estimate=surrogate_model_outputs.target_estimate[best_idx].item(),
            target_var=surrogate_model_outputs.target_var[best_idx].item(),
            probabilities=probabilities[best_idx].item(),
            expected_improvement=ei_value[best_idx].item(),
            success_probability=surrogate_model_outputs.success_probability[best_idx].item(),
        )
        return SuggestionInBasic(real_number_input=samples_in_basic[best_idx], log_info=log_info)

    def get_surrogate_model(self) -> SurrogateModel:
        params = SurrogateModelParams(
            real_dims=self.real_dim,
            better_direction_sign=self.params.better_direction_sign,
            outstanding_suggestion_estimator=self.params.outstanding_suggestion_estimator,
            device=self.device,
            scale_length=self.search_radius_in_basic.item(),
        )
        return SurrogateModel(params)

    def _get_random_suggestion(
        self, suggestion_id: Optional[str] = None, is_suggestion_remembered: bool = True
    ) -> SuggestOutput:
        suggestions: List[SuggestionInBasic] = []
        request_samples = 4
        while len(suggestions) < 1:
            suggestions = self.sample_search_space(request_samples)
            request_samples *= 2
            if request_samples > 8 * self.overgenerate_candidate_factor:
                raise Exception("Cannot get a random sample :(")
        return SuggestOutput(
            suggestion=self._basic_space_to_param_space_for_suggestion(
                suggestions[0], is_remembered=is_suggestion_remembered, suggestion_id=suggestion_id
            )
        )

    @logger.catch()
    def suggest(self, suggestion_id: Optional[str] = None, is_suggestion_remembered: bool = True) -> SuggestOutput:
        """
        Return a new suggestion

        Keyword arguments:
        suggestion_id: Optional[str] -- will be used to keep track of the suggestion
        is_suggestion_remembered: bool -- whether to store the suggestion, which be fit with a surrogate value until it is observed

        Returns:
        suggestion: ParamDictType -- Dictionary of parameters suggested
        log: Dict[str, Any] -- Additional details about the suggestion, such as predicted cost and output
        """
        if len(self.observations_in_basic) < self.params.num_random_samples:
            return self._get_random_suggestion(suggestion_id, is_suggestion_remembered)

        suggestion_in_basic: Optional[SuggestionInBasic]
        if (
            self.params.resample_frequency > 0
            and len(self.observations_in_basic) > (len(self.resample_history) + 1) * self.params.resample_frequency
        ):
            suggestion_in_basic = self._get_resample_suggestion()
            suggestion = self._basic_space_to_param_space_for_suggestion(
                suggestion_in_basic, is_remembered=is_suggestion_remembered, suggestion_id=suggestion_id
            )
            return SuggestOutput(suggestion=suggestion)

        try:
            suggestion_in_basic = self._generate_candidate()
        except Exception as e:
            logger.warning(f"Got error generating candidate {e}: {traceback.format_exc()}")
            suggestion_in_basic = None

        if suggestion_in_basic is None:
            logger.warning("No candidates found, choosing at random")
            return self._get_random_suggestion(suggestion_id, is_suggestion_remembered)

        suggestion_in_param = self._basic_space_to_param_space_for_suggestion(
            suggestion_in_basic, is_remembered=is_suggestion_remembered, suggestion_id=suggestion_id
        )
        log_dict = self._get_suggestion_log(suggestion_in_param, suggestion_in_basic)

        return SuggestOutput(suggestion=suggestion_in_param, log=log_dict)

    def forget_suggestion(self, suggestion_to_forget: ParamDictType) -> ParamDictType:
        """
        Removes suggestion from outstanding_suggestions
        """
        if SUGGESTION_ID_DICT_KEY in suggestion_to_forget:
            suggestion_index = suggestion_to_forget[SUGGESTION_ID_DICT_KEY]
            assert isinstance(suggestion_index, str)
            if suggestion_index == "":
                logger.warning("Empty suggestion uuid!")
            else:
                assert (
                    suggestion_index in self.outstanding_suggestions
                ), "Got an observation with suggestion uuid we don't recognize"
                del self.outstanding_suggestions[suggestion_index]

        return suggestion_to_forget

    def _observation_group_output_pos_better(self, group: Sequence[ObservationInBasic]):
        return observation_group_output(group) * self.params.better_direction_sign

    def _get_pareto_groups(self, is_conservative: bool = False) -> List[Tuple[ObservationInBasic, ...]]:
        grouped_observations = group_observations(self.observations_in_basic)
        if is_conservative:
            pareto_groups = get_pareto_groups_conservative(
                grouped_observations, self.params.min_pareto_cost_fraction, self.params.better_direction_sign
            )
        else:
            pareto_groups = get_pareto_groups(
                grouped_observations, self.params.min_pareto_cost_fraction, self.params.better_direction_sign
            )

        return pareto_groups

    def _get_pareto_set(self, is_conservative: bool = False) -> List[ObservationInBasic]:
        pareto_set: List[ObservationInBasic] = []
        for group in self._get_pareto_groups(is_conservative):
            pareto_set.extend(group)
        return pareto_set

    def _get_resample_suggestion(self) -> SuggestionInBasic:
        # we specifically want the non-conservative version so we can resample!
        pareto_groups = self._get_pareto_groups(is_conservative=False)

        # This ordering will determine which obs we select:
        # we select the first one which has the minimum number of observations/suggestions already
        pareto_groups.sort(key=observation_group_cost)

        pareto_obs_with_counts: List[Tuple[ObservationInBasic, int]] = []
        for group in pareto_groups:
            first_obs = group[0]
            count = len(group)
            for suggestion in self.outstanding_suggestions.values():
                if torch.all(torch.isclose(suggestion.real_number_input, first_obs.real_number_input)):
                    count += 1
            pareto_obs_with_counts.append((first_obs, count))

        min_count = min(x[1] for x in pareto_obs_with_counts)
        selected_obs = next(x[0] for x in pareto_obs_with_counts if x[1] == min_count)

        resample_suggestion = SuggestionInBasic(selected_obs.real_number_input)
        self.resample_history.append(resample_suggestion)
        return resample_suggestion

    def _init_wandb(self) -> None:
        if self.params.is_wandb_logging_enabled:
            wandb_params = self.params.wandb_params
            if wandb_params.run_id is None:
                self.wandb_run = wandb.init(
                    config=self.params.to_dict(),
                    project=wandb_params.project_name,
                    group=wandb_params.group_name,
                    name=wandb_params.run_name,
                    dir=wandb_params.root_dir,
                )
            else:
                self.wandb_run = wandb.init(
                    config=self.params.to_dict(),
                    project=wandb_params.project_name,
                    group=wandb_params.group_name,
                    name=wandb_params.run_name,
                    dir=wandb_params.root_dir,
                    reinit=True,
                    resume="allow",
                    id=wandb_params.run_id,
                )

    @property
    def cumulative_cost(self) -> float:
        return sum(x.cost for x in self.observations_in_basic)

    @property
    def observation_count(self) -> int:
        return len(self.observations_in_basic) + len(self.failed_observations_in_basic)

    def _get_observation_log(self, observation: ObservationInParam) -> Dict[str, Any]:
        log_dict: Dict[str, Any] = {
            "observation_count": self.observation_count,
            "cumulative_cost": self.cumulative_cost,
            "failed_observation_count": len(self.failed_observations_in_basic),
        }
        log_dict.update(add_dict_key_prefix(observation.input, "observation/"))
        log_dict["observation/is_failure"] = 1 if observation.is_failure else 0
        if not observation.is_failure:
            log_dict["observation/output"] = observation.output
            log_dict["observation/cost"] = observation.cost

        if len(self.observations_in_basic) > 0:
            best_observation_in_basic = max(
                self.observations_in_basic, key=lambda x: x.output * self.params.better_direction_sign
            )
            best_observation_in_param = self._basic_space_to_param_space(best_observation_in_basic.real_number_input)
            log_dict.update(add_dict_key_prefix(best_observation_in_param, "best_observation/"))

            log_dict["best_observation/output"] = best_observation_in_basic.output
            log_dict["best_observation/cost"] = best_observation_in_basic.cost

            grouped_observations = group_observations(self.observations_in_basic)
            pareto_groups = self._get_pareto_groups(self.params.is_pareto_group_selection_conservative)
            log_dict["pareto_group_count"] = len(pareto_groups)
            log_dict["pareto_area"] = pareto_area_from_groups(pareto_groups)

            resampled_groups = [x for x in grouped_observations if len(x) > 1]
            if len(resampled_groups) > 0:
                best_resampled_observations = max(
                    resampled_groups, key=lambda x: observation_group_output(x) * self.params.better_direction_sign
                )
                resampled_observation_in_param = self._basic_space_to_param_space(
                    best_resampled_observations[0].real_number_input
                )
                best_resampled_observation_outputs = [x.output for x in best_resampled_observations]
                log_dict.update(add_dict_key_prefix(resampled_observation_in_param, "best_resampled_observation/"))

                log_dict["best_resampled_observation/output_mean"] = np.mean(best_resampled_observation_outputs)
                log_dict["best_resampled_observation/output_std_dev"] = np.std(best_resampled_observation_outputs)
                log_dict["best_resampled_observation/cost_mean"] = observation_group_cost(best_resampled_observations)
                log_dict["best_resampled_observation/sample_count"] = len(best_resampled_observations)

            if self.params.is_wandb_logging_enabled and len(pareto_groups) > 0:
                plot_path = get_pareto_curve_plot(
                    self.observations_in_basic,
                    pareto_groups,
                    self.params.wandb_params.root_dir,
                    obs_count=self.observation_count,
                )
                if plot_path is not None:
                    log_dict["pareto_curve"] = wandb.Image(plot_path)

        if self.params.is_wandb_logging_enabled and self.params.wandb_params.is_observation_logged and self.wandb_run:
            wandb.log(log_dict)

        return log_dict

    def _get_suggestion_log(
        self, suggestion: ParamDictType, observation_in_surrogate: SuggestionInBasic
    ) -> Dict[str, Any]:
        log_dict: Dict[str, Any] = {
            "observation_count": self.observation_count,
            "cumulative_cost": self.cumulative_cost,
            "failed_observation_count": len(self.failed_observations_in_basic),
        }
        log_dict.update(add_dict_key_prefix(suggestion, "suggestion/"))
        log_dict.update(add_dict_key_prefix(observation_in_surrogate.log_info, "suggestion_meta/"))
        if self.params.is_wandb_logging_enabled and self.params.wandb_params.is_suggestion_logged and self.wandb_run:
            wandb.log(log_dict)

        return log_dict

    def _autosave(self) -> None:
        filename = f"bones_{self.observation_count}obs.pt"
        self.save_to_file(filename)
        if self.wandb_run:
            wandb.save(filename)

    @staticmethod
    def load_from_file(filename: str, is_wandb_logging_enabled: bool = False) -> "CARBS":
        optimizer = cast(CARBS, torch.load(filename))
        if is_wandb_logging_enabled:
            optimizer._init_wandb()
        else:
            try:
                optimizer.params = evolve(optimizer.params, is_wandb_logging_enabled=False)
            except AttributeError as e:
                logger.warning(f"Skipping disabling wandb logging due to AttributeError {e}")
        return optimizer

    @classmethod
    def load_from_string(
        cls, state: str, intermediate_filename: str = "bones_state.pt", is_wandb_logging_enabled: bool = False
    ) -> "CARBS":
        binary_data = base64.b64decode(state)
        with open(intermediate_filename, "wb") as file:
            file.write(binary_data)
        return cls.load_from_file(intermediate_filename, is_wandb_logging_enabled)

    def save_to_file(self, filename: str) -> None:
        torch.save(self, filename)

    def serialize(self, intermediate_filename: str = "bones_state.pt") -> str:
        self.save_to_file(intermediate_filename)
        return base64.b64encode(open(intermediate_filename, "rb").read()).decode("utf-8")

    def warm_start_from_wandb(
        self,
        run_name: str,
        is_prior_observation_valid: bool = False,
        added_parameters: Optional[ParamDictType] = None,
    ) -> None:
        filename = load_latest_checkpoint_from_wandb_run(run_name)
        self.warm_start(filename, is_prior_observation_valid, added_parameters)

    def warm_start(
        self,
        filename: str,
        is_prior_observation_valid: bool = False,
        added_parameters: Optional[ParamDictType] = None,
    ) -> None:
        prior_bones = CARBS.load_from_file(filename)

        # Copying over observations
        if is_prior_observation_valid:
            prior_keys = set(prior_bones.param_space_by_name.keys())
            current_keys = set(self.param_space_by_name.keys())
            added_keys = set() if added_parameters is None else set(added_parameters.keys())
            assert_empty(prior_keys - current_keys, "Prior observations used params that are not included in search")
            assert_empty(current_keys - (prior_keys | added_keys), "Pass in added parameters")
            assert_empty(added_keys & prior_keys, "Cannot add param already in prior")

            for prior_observation_in_basic in prior_bones.observations_in_basic:
                prior_observation_in_params = prior_bones._basic_space_to_param_space(
                    prior_observation_in_basic.real_number_input
                )
                if added_parameters is not None:
                    prior_observation_in_params.update(added_parameters)

                try:
                    self._add_observation(
                        ObservationInParam(
                            input=prior_observation_in_params,
                            output=prior_observation_in_basic.output,
                            cost=prior_observation_in_basic.cost,
                        )
                    )
                except ValueError as e:
                    logger.warning(
                        f"Observation {prior_observation_in_params} not valid in current space: {e}; skipping"
                    )

            logger.info(f"Loaded {len(self.observations_in_basic)} observations from prior run")

            for prior_observation_in_basic in prior_bones.failed_observations_in_basic:
                prior_observation_in_params = prior_bones._basic_space_to_param_space(
                    prior_observation_in_basic.real_number_input
                )
                if added_parameters is not None:
                    prior_observation_in_params.update(added_parameters)

                try:
                    self._add_observation(
                        ObservationInParam(
                            input=prior_observation_in_params,
                            output=prior_observation_in_basic.output,
                            cost=prior_observation_in_basic.cost,
                            is_failure=True,
                        )
                    )
                except ValueError as e:
                    logger.warning(
                        f"Observation {prior_observation_in_params} not valid in current space: {e}; skipping"
                    )
            for prior_resampled_observation in prior_bones.resample_history:
                prior_observation_in_params = prior_bones._basic_space_to_param_space(
                    prior_resampled_observation.real_number_input
                )
                if added_parameters is not None:
                    prior_observation_in_params.update(added_parameters)
                prior_real = self._param_space_real_to_basic_space_real(prior_observation_in_params)
                self.resample_history.append(
                    SuggestionInBasic(
                        real_number_input=prior_real,
                        log_info=prior_resampled_observation.log_info,
                    )
                )

            logger.info(f"Loaded {len(self.failed_observations_in_basic)} failed observations from prior run")

        # Real number space overlap
        prior_real_idx_lookup = {name: idx for idx, name in enumerate(prior_bones._real_number_space_by_name)}
        shared_real_params = [
            (name, idx, prior_real_idx_lookup[name])
            for idx, name in enumerate(self._real_number_space_by_name.keys())
            if name in prior_real_idx_lookup
        ]

        if added_parameters is not None:
            for param_name, param_value in added_parameters.items():
                if param_name in self._real_number_space_by_name:
                    param_idx = ordered_dict_index(self._real_number_space_by_name, param_name)
                    param_value_in_basic = self._real_number_space_by_name[param_name].basic_from_param(param_value)
                    self.search_center_in_basic[param_idx] = param_value_in_basic
