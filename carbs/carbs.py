# %%
import base64
import io
import math
import os
import random
import threading
import traceback
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import cast

import numpy as np
import torch
import wandb
from attr import evolve
from loguru import logger
from torch import Tensor
from torch.distributions import Categorical
from torch.distributions import Distribution
from torch.distributions import Normal
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

from carbs.model import SurrogateModel
from carbs.utils import CARBSParams, load_latest_checkpoint_from_wandb_run
from carbs.utils import CARBS_CHECKPOINT_PREFIX
from carbs.utils import CARBS_CHECKPOINT_SUFFIX
from carbs.utils import ObservationGroup
from carbs.utils import ObservationInBasic
from carbs.utils import ObservationInParam
from carbs.utils import ObserveOutput
from carbs.utils import Param
from carbs.utils import ParamDictType
from carbs.utils import RealNumberSpace
from carbs.utils import SUGGESTION_ID_DICT_KEY
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
from carbs.utils import observation_group_cost
from carbs.utils import observation_group_output
from carbs.utils import ordered_dict_index
from carbs.utils import pareto_area_from_groups


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

    def __init__(self, config: CARBSParams, params: List[Param]) -> None:
        logger.info(f"Running CARBS with config {config}")
        self.config = config
        self.params = params
        experiment_name = os.environ.get(
            "EXPERIMENT_NAME", config.wandb_params.run_name
        )
        self.experiment_name = (
            experiment_name if experiment_name is not None else "carbs_experiment"
        )
        self.param_space_by_name = {param.name: param.space for param in params}
        self._real_number_space_by_name = OrderedDict(
            (k, dim)
            for k, dim in self.param_space_by_name.items()
            if isinstance(dim, RealNumberSpace)
        )
        assert len(self._real_number_space_by_name) == len(
            self.param_space_by_name
        ), "Real numbers only supported now"
        self.real_dim = len(self._real_number_space_by_name)
        self.search_center_in_basic = torch.zeros(
            (self.real_dim,)
        )  # Immediately overwritten in set_search_center
        self.search_radius_in_basic = torch.tensor(
            float(self.config.initial_search_radius)
        )
        self.set_search_center({param.name: param.search_center for param in params})

        self.success_observations: List[ObservationInBasic] = []
        self.failure_observations: List[ObservationInBasic] = []

        self.min_bounds_in_basic = torch.tensor(
            [
                dim.basic_from_param(dim.min_bound)
                for dim in self._real_number_space_by_name.values()
            ]
        )
        self.max_bounds_in_basic = torch.tensor(
            [
                dim.basic_from_param(dim.max_bound)
                for dim in self._real_number_space_by_name.values()
            ]
        )
        self._suggest_or_observe_lock = threading.Lock()
        self.outstanding_suggestions: Dict[str, SuggestionInBasic] = (
            {}
        )  # Keys are UUIDs

        self._set_seed(self.config.seed)

        # Only used so far to keep track of how many resamples we have suggested
        self.resample_count: int = 0

        num_dims_with_bounds = sum(
            (not math.isinf(dim.min_bound)) or (not math.isinf(dim.max_bound))
            for dim in self._real_number_space_by_name.values()
        )
        if num_dims_with_bounds > 10:
            logger.info(
                f"Many dimensions with bounds ({num_dims_with_bounds}), sampling may be slow"
            )
        self.overgenerate_candidate_factor = 2 ** (num_dims_with_bounds // 2)

        self.wandb_run: Union[Run, RunDisabled, None] = None
        self._init_wandb()

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = f"cuda:{torch.cuda.current_device()}"

    def set_search_center(self, input_in_param: ParamDictType) -> None:
        self.search_center_in_basic = self._param_space_real_to_basic_space_real(
            input_in_param
        )

    def suggest(
        self, suggestion_id: Optional[str] = None, is_suggestion_remembered: bool = True
    ) -> SuggestOutput:
        """
        Return a new suggestion

        Keyword arguments:
        suggestion_id: Optional[str] -- will be used to keep track of the suggestion
        is_suggestion_remembered: bool -- whether to store the suggestion, which be fit with a surrogate value until it is observed

        Returns:
        SuggestOutput, which contains:
        suggestion: ParamDictType -- Dictionary of parameters suggested
        log: Dict[str, Any] -- Additional details about the suggestion, such as predicted cost and output
        """
        with self._suggest_or_observe_lock:
            if self._is_random_sampling():
                return self._get_random_suggestion(
                    suggestion_id, is_suggestion_remembered
                )

            suggestion_in_basic: Optional[SuggestionInBasic]
            if (
                self.config.resample_frequency > 0
                and len(self.success_observations)
                > (self.resample_count + 1) * self.config.resample_frequency
            ):
                suggestion_in_basic = self._get_resample_suggestion()
                suggestion_in_param = self._basic_space_to_param_space(
                    suggestion_in_basic.real_number_input
                )
                if is_suggestion_remembered:
                    self._remember_suggestion(
                        suggestion_in_param,
                        suggestion_in_basic,
                        suggestion_id=suggestion_id,
                    )
                return SuggestOutput(suggestion=suggestion_in_param)

            try:
                suggestion_in_basic = self._generate_candidate()
            except Exception as e:
                logger.warning(
                    f"Got error generating candidate {e}: {traceback.format_exc()}"
                )
                suggestion_in_basic = None

            if suggestion_in_basic is None:
                logger.warning("No candidates found, choosing at random")
                return self._get_random_suggestion(
                    suggestion_id, is_suggestion_remembered
                )

            suggestion_in_param = self._basic_space_to_param_space(
                suggestion_in_basic.real_number_input
            )
            if is_suggestion_remembered:
                self._remember_suggestion(
                    suggestion_in_param,
                    suggestion_in_basic,
                    suggestion_id=suggestion_id,
                )
            log_dict = self._get_suggestion_log(
                suggestion_in_param, suggestion_in_basic
            )

            return SuggestOutput(suggestion=suggestion_in_param, log=log_dict)

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
        with self._suggest_or_observe_lock:
            self.forget_suggestion(new_observation_in_param.input)
            self._add_observation(new_observation_in_param)
            logs = self._get_observation_log(new_observation_in_param)

            if self.config.is_saved_on_every_observation:
                self._autosave()

            return ObserveOutput(logs=logs)

    def forget_suggestion(self, suggestion_to_forget: ParamDictType) -> None:
        """
        Removes suggestion from outstanding_suggestions
        """
        if SUGGESTION_ID_DICT_KEY in suggestion_to_forget:
            suggestion_index = suggestion_to_forget[SUGGESTION_ID_DICT_KEY]
            assert isinstance(suggestion_index, str)
            if suggestion_index in self.outstanding_suggestions:
                del self.outstanding_suggestions[suggestion_index]
            else:
                logger.info(f"Got unrecognized suggestion uuid `{suggestion_index}`")
        else:
            pass  # It's fine to forget a suggestion without a UUID, but it doesn't do anything, we weren't remembering it anyway.

    def initialize_from_observations(
        self, observations: List[ObservationInParam]
    ) -> ObserveOutput:
        logs: Dict[str, Any] = {}
        for observation in observations:
            self._add_observation(observation)
            # logs.update(self._get_observation_log(observation))

        if len(observations) > 0:
            best_obs = max(
                observations, key=lambda x: x.output * self.config.better_direction_sign
            )
            self.set_search_center(best_obs.input)
        else:
            logger.warning("No way to set the search center!")

        return ObserveOutput(logs=logs)

    def __getstate__(self) -> Dict[str, object]:
        state = self.__dict__.copy()
        state["wandb_run"] = None
        state.pop("_suggest_or_observe_lock")
        return state

    def __setstate__(self, state: Dict[str, object]) -> None:
        self.__dict__.update(state)
        self._suggest_or_observe_lock = threading.Lock()

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _get_mask_for_invalid_points_in_basic(self, input_in_basic: Tensor) -> Tensor:
        # mypy understand these types but pycharm does not :'(
        is_above_min_bounds = aggregate_logical_and_across_dim(
            input_in_basic > self.min_bounds_in_basic.unsqueeze(0)
        )
        is_below_max_bounds = aggregate_logical_and_across_dim(
            input_in_basic < self.max_bounds_in_basic.unsqueeze(0)
        )
        mask = torch.logical_and(is_above_min_bounds, is_below_max_bounds)
        return mask

    def _round_integer_values_in_basic(self, input_in_basic: Tensor) -> Tensor:
        for idx, space in enumerate(self._real_number_space_by_name.values()):
            input_in_basic[..., idx] = space.round_tensor_in_basic(
                input_in_basic[..., idx]
            )
        return input_in_basic

    def _param_space_real_to_basic_space_real(
        self, input_in_param: ParamDictType
    ) -> Tensor:
        return torch.tensor(
            [
                dim.basic_from_param(input_in_param[k])
                for k, dim in self._real_number_space_by_name.items()
            ]
        )

    def _param_space_obs_to_basic_space_obs(
        self, observation_in_params: ObservationInParam
    ) -> ObservationInBasic:
        input_in_param = observation_in_params.input
        suggestion_id = input_in_param.pop(SUGGESTION_ID_DICT_KEY, None)
        real_number_input = self._param_space_real_to_basic_space_real(input_in_param)
        return ObservationInBasic(
            real_number_input=real_number_input,
            output=float(observation_in_params.output),
            cost=float(observation_in_params.cost),
            suggestion_id=str(suggestion_id) if suggestion_id else None,
        )

    def _basic_space_to_param_space(self, real_number_input: Tensor) -> ParamDictType:
        return {
            k: dim.param_from_basic(float(v))
            for (k, dim), v in zip(
                self._real_number_space_by_name.items(), real_number_input
            )
        }

    def _basic_space_to_unrounded_param_space(
        self, real_number_input: Tensor
    ) -> ParamDictType:
        return {
            k: dim.param_from_basic(float(v), is_rounded=False)
            for (k, dim), v in zip(
                self._real_number_space_by_name.items(), real_number_input
            )
        }

    def _remember_suggestion(
        self,
        suggestion_in_param: ParamDictType,
        suggestion_in_basic: SuggestionInBasic,
        suggestion_id: Optional[str],
    ) -> ParamDictType:
        if suggestion_id is None:
            suggestion_id = str(uuid.uuid4())
        suggestion_in_param[SUGGESTION_ID_DICT_KEY] = suggestion_id
        self.outstanding_suggestions[suggestion_id] = suggestion_in_basic
        return suggestion_in_param

    def _add_observation(
        self, observation_in_param: ObservationInParam
    ) -> Optional[ObservationInBasic]:
        observation_in_basic = self._param_space_obs_to_basic_space_obs(
            observation_in_param
        )
        if (
            observation_in_param.is_failure
            or not np.isfinite(observation_in_basic.output)
            or np.isnan(observation_in_basic.output)
        ):
            self.failure_observations.append(observation_in_basic)
            return None

        self.success_observations.append(observation_in_basic)
        return observation_in_basic

    @property
    def _search_distribution_in_basic(self) -> Distribution:
        return Normal(0, self.search_radius_in_basic)

    def _sample_around_origins_in_basic(
        self, num_samples: int, origins_in_basic: Tensor
    ) -> Tuple[Tensor, Tensor]:
        assert (
            len(origins_in_basic.shape) == 2
            and origins_in_basic.shape[1] == self.real_dim
        ), f"Bad shape for origins_in_natural: {origins_in_basic.shape}"
        origin_index = Categorical(
            logits=torch.zeros((origins_in_basic.shape[0],))
        ).sample(torch.Size((num_samples,)))
        origin_samples = origins_in_basic[origin_index]
        real_samples: Tensor = (
            origin_samples
            + self._search_distribution_in_basic.sample(
                torch.Size((num_samples, self.real_dim))
            )
        )
        probabilities = self._get_probability_in_search_space(
            input_in_basic=real_samples, origins_in_basic=origins_in_basic
        )
        return real_samples, probabilities

    def _get_probability_in_search_space(
        self,
        input_in_basic: Tensor,
        origins_in_basic: Tensor,
    ) -> Tensor:
        input_distance = torch.norm(
            input_in_basic.unsqueeze(1) - origins_in_basic.unsqueeze(0), dim=-1
        ).to(self.device)
        real_relative_probability_per_origin = torch.exp(
            self._search_distribution_in_basic.log_prob(input_distance)
        )
        # we take the max probability origin
        real_relative_probability: Tensor = real_relative_probability_per_origin.max(
            dim=-1
        ).values
        return real_relative_probability

    def _is_random_sampling(self) -> bool:
        # Random sampling as opposed to Bayesian optimization.
        return len(self.success_observations) < self.config.num_random_samples

    def sample_search_space(self, num_samples: int) -> List[SuggestionInBasic]:
        # The length of the list returned will be less than or equal to num_samples. Filtering is done by _get_mask_for_invalid_points_in_basic, which checks min and max values.
        if self._is_random_sampling():
            # Main case
            origins_in_basic = self.search_center_in_basic.unsqueeze(0)
        else:
            # Edge case that only occurs as a fallback when we can't generate a candidate
            pareto_groups = self._get_pareto_groups(
                is_conservative=self.config.is_pareto_group_selection_conservative
            )
            origins_in_basic = torch.stack(
                [x[0].real_number_input for x in pareto_groups], dim=0
            )

        samples_in_basic, _ = self._sample_around_origins_in_basic(
            num_samples * self.overgenerate_candidate_factor, origins_in_basic
        )

        valid_sample_mask = self._get_mask_for_invalid_points_in_basic(samples_in_basic)
        samples_in_basic = samples_in_basic[valid_sample_mask][:num_samples]

        if samples_in_basic.shape[0] < num_samples:
            logger.warning(
                f"Undergenerated valid samples, requested {num_samples} generated {samples_in_basic.shape[0]}"
            )
            self._crank_oversampling_up()

        suggestions_in_basic = [
            SuggestionInBasic(real_number_input=x) for x in samples_in_basic
        ]

        return suggestions_in_basic

    @torch.no_grad()
    def _generate_candidate(self) -> Optional[SuggestionInBasic]:
        surrogate_model = self.get_surrogate_model()
        surrogate_model.fit_observations(self.success_observations)
        surrogate_model.fit_suggestions(list(self.outstanding_suggestions.values()))
        surrogate_model.fit_failures(
            self.success_observations, self.failure_observations
        )
        pareto_groups = self._get_pareto_groups(
            is_conservative=self.config.is_pareto_group_selection_conservative
        )
        all_pareto_observations = [obs for group in pareto_groups for obs in group]
        surrogate_model.fit_pareto_set(all_pareto_observations)

        num_samples_to_generate = (
            self.config.num_candidates_for_suggestion_per_dim * self.real_dim
        )

        origins_in_basic = torch.stack(
            [x[0].real_number_input for x in pareto_groups], dim=0
        )
        samples_in_basic, probabilities = self._sample_around_origins_in_basic(
            num_samples_to_generate * self.overgenerate_candidate_factor,
            origins_in_basic,
        )

        # do rounding first
        samples_in_basic = self._round_integer_values_in_basic(samples_in_basic)

        assert samples_in_basic.shape[0] > 0

        # This is why we overgenerated
        valid_sample_mask = self._get_mask_for_invalid_points_in_basic(samples_in_basic)
        samples_in_basic = samples_in_basic[valid_sample_mask][:num_samples_to_generate]
        probabilities = probabilities[valid_sample_mask][:num_samples_to_generate]

        if samples_in_basic.shape[0] < num_samples_to_generate:
            logger.info(
                f"Undergenerated valid samples, requested {num_samples_to_generate} generated {samples_in_basic.shape[0]}"
            )
            self._crank_oversampling_up()
        if samples_in_basic.shape[0] == 0:
            return None

        surrogate_model_outputs = surrogate_model.observe_surrogate(samples_in_basic)
        assert surrogate_model_outputs.pareto_surrogate is not None
        assert surrogate_model_outputs.pareto_estimate is not None

        pareto_surrogate = surrogate_model_outputs.pareto_surrogate
        if self.config.is_expected_improvement_pareto_value_clamped:
            # Biasing technique to encourage higher cost exploration: Choose a random cost along pareto front, and make the pareto value there the minimum for all observations

            # The pareto_groups are sorted, so the 0th and -1th elements represent the low and high values
            log_min_cost = math.log(observation_group_cost(pareto_groups[0]))
            if len(pareto_groups) > 1:
                log_max_cost = math.log(observation_group_cost(pareto_groups[-1]))
            else:
                log_max_cost = log_min_cost
            if self.config.max_suggestion_cost is not None:
                log_max_cost = min(
                    log_max_cost, math.log(self.config.max_suggestion_cost)
                )

            # Choose a random cost along the curve
            cost_threshold = math.exp(random.uniform(log_min_cost, log_max_cost))

            # Choose the surrogate value at that cost
            pareto_surrogate_at_threshold = (
                surrogate_model.get_pareto_surrogate_for_cost(cost_threshold)
            )

            # Modify the pareto surrogate to be at least that value. This effectively cuts off exploration of low expected value points, which biases the search toward the right side of the pareto curve.
            pareto_surrogate = torch.clamp(
                pareto_surrogate, min=pareto_surrogate_at_threshold
            )

        if self.config.is_expected_improvement_value_always_max:
            # NB: Only used for ablations
            best_pareto_group_output = observation_group_output(pareto_groups[-1])
            best_output_in_surrogate = surrogate_model._target_to_surrogate(
                torch.tensor([best_pareto_group_output])
            ).item()
            pareto_surrogate = (
                torch.ones_like(pareto_surrogate) * best_output_in_surrogate
            )

        max_cost_masking = torch.ones_like(surrogate_model_outputs.cost_estimate)
        if self.config.max_suggestion_cost is not None:
            max_cost_masking = cast(
                torch.BoolTensor,
                surrogate_model_outputs.cost_estimate < self.config.max_suggestion_cost,
            ).to(torch.float)
            assert (
                max_cost_masking.max().item() > 0.5
            ), f"No candidates below max cost bound {self.config.max_suggestion_cost}"

        ei_value = expected_improvement(
            surrogate_model_outputs.surrogate_output,
            surrogate_model_outputs.surrogate_var,
            best_mu=pareto_surrogate,
            exploration_bias=self.config.exploration_bias,
        )

        # Central equation: bias points by expected improvement, prior probability, and success probability, excluding points that are too expensive
        acquisition_function_value = (
            ei_value
            * probabilities
            * surrogate_model_outputs.success_probability
            * max_cost_masking
        )
        best_idx = int(torch.argmax(acquisition_function_value).item())

        # A single point is chosen by that argmax, so log the info for that point
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
            success_probability=surrogate_model_outputs.success_probability[
                best_idx
            ].item(),
        )
        return SuggestionInBasic(
            real_number_input=samples_in_basic[best_idx], log_info=log_info
        )

    def get_surrogate_model(self) -> SurrogateModel:
        params = SurrogateModelParams(
            real_dims=self.real_dim,
            better_direction_sign=self.config.better_direction_sign,
            outstanding_suggestion_estimator=self.config.outstanding_suggestion_estimator,
            device=self.device,
            scale_length=self.search_radius_in_basic.item(),
        )
        return SurrogateModel(params)

    def _get_random_suggestion(
        self,
        suggestion_id: Optional[str] = None,
        is_suggestion_remembered: bool = True,
        num_sampling_attempts: int = 8,
    ) -> SuggestOutput:
        for i in range(num_sampling_attempts):
            # Might as well try to grab 32 points since we get some parallelism for free. sample_search_space will crank up oversampling if needed.
            suggestions = self.sample_search_space(32)
            if len(suggestions) > 0:
                suggestion_in_param = self._basic_space_to_param_space(
                    suggestions[0].real_number_input
                )
                if is_suggestion_remembered:
                    self._remember_suggestion(
                        suggestion_in_param, suggestions[0], suggestion_id=suggestion_id
                    )
                return SuggestOutput(suggestion=suggestion_in_param)
        raise Exception("Cannot get a random sample :(")

    def _observation_group_output_pos_better(self, group: Sequence[ObservationInBasic]):
        return observation_group_output(group) * self.config.better_direction_sign

    def _crank_oversampling_up(self) -> None:
        self.overgenerate_candidate_factor = min(
            self.overgenerate_candidate_factor * 2, 2**self.real_dim
        )

    def _get_pareto_groups(
        self, is_conservative: bool = False
    ) -> Tuple[ObservationGroup, ...]:
        """
        Get pareto groups from success observations.

        :param is_conservative: see get_pareto_groups_conservative for description
        :return:
        """
        grouped_observations = group_observations(self.success_observations)
        if is_conservative:
            pareto_groups = get_pareto_groups_conservative(
                grouped_observations,
                self.config.min_pareto_cost_fraction,
                self.config.better_direction_sign,
            )
        else:
            pareto_groups = get_pareto_groups(
                grouped_observations,
                self.config.min_pareto_cost_fraction,
                self.config.better_direction_sign,
            )

        return pareto_groups

    def _get_pareto_set(
        self, is_conservative: bool = False
    ) -> Tuple[ObservationInBasic, ...]:
        pareto_set: List[ObservationInBasic] = []
        for group in self._get_pareto_groups(is_conservative):
            pareto_set.extend(group)
        return tuple(pareto_set)

    def _get_resample_suggestion(self) -> SuggestionInBasic:
        # we specifically want the non-conservative version so we can resample!
        pareto_groups = list(self._get_pareto_groups(is_conservative=False))

        # This ordering will determine which obs we select:
        # we select the first one which has the minimum number of observations/suggestions already
        pareto_groups.sort(key=observation_group_cost)

        pareto_obs_with_counts: List[Tuple[ObservationInBasic, int]] = []
        for group in pareto_groups:
            # Take the first obs because they all have the same sample point
            first_obs = group[0]
            count = len(group)
            # Add to the count of the group the number of outstanding suggestions that are already out to resample this group.
            for suggestion in self.outstanding_suggestions.values():
                if torch.all(
                    torch.isclose(
                        suggestion.real_number_input, first_obs.real_number_input
                    )
                ):
                    count += 1
            pareto_obs_with_counts.append((first_obs, count))

        # Resample one of the observation groups with the minimum count along the pareto front
        min_count = min(x[1] for x in pareto_obs_with_counts)
        selected_obs = next(x[0] for x in pareto_obs_with_counts if x[1] == min_count)

        resample_suggestion = SuggestionInBasic(selected_obs.real_number_input)
        self.resample_count += 1
        return resample_suggestion

    def _init_wandb(self) -> None:
        if self.config.is_wandb_logging_enabled:
            wandb_params = self.config.wandb_params
            if wandb_params.run_id is None:
                self.wandb_run = wandb.init(
                    config=self.config.to_dict(),
                    project=wandb_params.project_name,
                    group=wandb_params.group_name,
                    name=wandb_params.run_name,
                    dir=wandb_params.root_dir,
                )
            else:
                self.wandb_run = wandb.init(
                    config=self.config.to_dict(),
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
        # We can't add cost of failed observations, they may not be valid
        return sum(x.cost for x in self.success_observations)

    @property
    def observation_count(self) -> int:
        return len(self.success_observations) + len(self.failure_observations)

    def _get_observation_log(self, observation: ObservationInParam) -> Dict[str, Any]:
        log_dict: Dict[str, Any] = {
            "observation_count": self.observation_count,
            "cumulative_cost": self.cumulative_cost,
            "failed_observation_count": len(self.failure_observations),
        }
        log_dict.update(add_dict_key_prefix(observation.input, "observation/"))
        log_dict["observation/is_failure"] = 1 if observation.is_failure else 0
        if not observation.is_failure:
            log_dict["observation/output"] = observation.output
            log_dict["observation/cost"] = observation.cost

        if len(self.success_observations) > 0:
            best_observation_in_basic = max(
                self.success_observations,
                key=lambda x: x.output * self.config.better_direction_sign,
            )
            best_observation_in_param = self._basic_space_to_param_space(
                best_observation_in_basic.real_number_input
            )
            log_dict.update(
                add_dict_key_prefix(best_observation_in_param, "best_observation/")
            )

            log_dict["best_observation/output"] = best_observation_in_basic.output
            log_dict["best_observation/cost"] = best_observation_in_basic.cost

            grouped_observations = group_observations(self.success_observations)
            pareto_groups = self._get_pareto_groups(
                self.config.is_pareto_group_selection_conservative
            )
            log_dict["pareto_group_count"] = len(pareto_groups)
            log_dict["pareto_area"] = pareto_area_from_groups(pareto_groups)

            resampled_groups = [x for x in grouped_observations if len(x) > 1]
            if len(resampled_groups) > 0:
                best_resampled_observations = max(
                    resampled_groups,
                    key=lambda x: observation_group_output(x)
                    * self.config.better_direction_sign,
                )
                resampled_observation_in_param = self._basic_space_to_param_space(
                    best_resampled_observations[0].real_number_input
                )
                best_resampled_observation_outputs = [
                    x.output for x in best_resampled_observations
                ]
                log_dict.update(
                    add_dict_key_prefix(
                        resampled_observation_in_param, "best_resampled_observation/"
                    )
                )

                log_dict["best_resampled_observation/output_mean"] = np.mean(
                    best_resampled_observation_outputs
                )
                log_dict["best_resampled_observation/output_std_dev"] = np.std(
                    best_resampled_observation_outputs
                )
                log_dict["best_resampled_observation/cost_mean"] = (
                    observation_group_cost(best_resampled_observations)
                )
                log_dict["best_resampled_observation/sample_count"] = len(
                    best_resampled_observations
                )

            if self.config.is_wandb_logging_enabled and len(pareto_groups) > 0:
                plot_path = get_pareto_curve_plot(
                    self.success_observations,
                    pareto_groups,
                    self.config.wandb_params.root_dir,
                    obs_count=self.observation_count,
                )
                if plot_path is not None:
                    log_dict["pareto_curve"] = wandb.Image(plot_path)

        if (
            self.config.is_wandb_logging_enabled
            and self.config.wandb_params.is_observation_logged
            and self.wandb_run
        ):
            wandb.log(log_dict)

        return log_dict

    def _get_suggestion_log(
        self, suggestion: ParamDictType, observation_in_surrogate: SuggestionInBasic
    ) -> Dict[str, Any]:
        log_dict: Dict[str, Any] = {
            "observation_count": self.observation_count,
            "cumulative_cost": self.cumulative_cost,
            "failed_observation_count": len(self.failure_observations),
        }
        log_dict.update(add_dict_key_prefix(suggestion, "suggestion/"))
        log_dict.update(
            add_dict_key_prefix(observation_in_surrogate.log_info, "suggestion_meta/")
        )
        if (
            self.config.is_wandb_logging_enabled
            and self.config.wandb_params.is_suggestion_logged
            and self.wandb_run
        ):
            wandb.log(log_dict)

        return log_dict

    def _autosave(self) -> None:
        filename = f"{CARBS_CHECKPOINT_PREFIX}{self.observation_count}{CARBS_CHECKPOINT_SUFFIX}"
        self.save_to_file(
            filename, upload_to_wandb=self.config.is_wandb_logging_enabled
        )

    @staticmethod
    def load_from_file(
        f: Union[str, io.BytesIO],
        is_wandb_logging_enabled: bool = False,
        override_params: Optional[CARBSParams] = None,
    ) -> "CARBS":
        state = torch.load(f)
        if override_params is not None:
            state["config"] = override_params
        if not is_wandb_logging_enabled:
            state["config"] = evolve(state["config"], is_wandb_logging_enabled=False)
        optimizer = CARBS.load_state_dict(state)
        return optimizer

    @classmethod
    def load_from_string(
        cls, state: str, is_wandb_logging_enabled: bool = False
    ) -> "CARBS":
        return cls.load_from_file(
            io.BytesIO(base64.b64decode(state)), is_wandb_logging_enabled
        )

    def get_state_dict(self) -> Dict[str, Any]:
        outstanding_suggestions: Dict[str, SuggestionInBasic] = {}
        for key, suggestion in self.outstanding_suggestions.items():
            # TODO: don't carry around these enormous real_number_inputs in the first place
            outstanding_suggestions[key] = evolve(
                suggestion, real_number_input=suggestion.real_number_input.clone()
            )
        return {
            "config": self.config,
            "params": self.params,
            "success_observations": self.success_observations,
            "failure_observations": self.failure_observations,
            "outstanding_suggestions": outstanding_suggestions,
            "resample_count": self.resample_count,
        }

    @classmethod
    def load_state_dict(cls, state: Dict[str, Any]) -> "CARBS":
        optimizer = CARBS(state["config"], state["params"])
        optimizer.success_observations = state["success_observations"]
        optimizer.failure_observations = state["failure_observations"]
        optimizer.outstanding_suggestions = state["outstanding_suggestions"]
        optimizer.resample_count = state["resample_count"]
        return optimizer

    def save_to_file(self, filename: str, upload_to_wandb: bool = False) -> None:
        checkpoint_path = (
            Path(self.config.checkpoint_dir) / self.experiment_name / filename
        )
        checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(self.get_state_dict(), checkpoint_path)
        if upload_to_wandb:
            wandb.save(checkpoint_path)

    def serialize(self) -> str:
        buf = io.BytesIO()
        torch.save(self.get_state_dict(), buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

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
        prior_carbs = CARBS.load_from_file(filename)

        # Copying over observations
        if is_prior_observation_valid:
            prior_keys = set(prior_carbs.param_space_by_name.keys())
            current_keys = set(self.param_space_by_name.keys())
            added_keys = (
                set() if added_parameters is None else set(added_parameters.keys())
            )
            assert_empty(
                prior_keys - current_keys,
                "Prior observations used params that are not included in search",
            )
            assert_empty(
                current_keys - (prior_keys | added_keys), "Pass in added parameters"
            )
            assert_empty(added_keys & prior_keys, "Cannot add param already in prior")

            for prior_observation_in_basic in prior_carbs.success_observations:
                prior_observation_in_params = prior_carbs._basic_space_to_param_space(
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

            logger.info(
                f"Loaded {len(self.success_observations)} observations from prior run"
            )

            for prior_observation_in_basic in prior_carbs.failure_observations:
                prior_observation_in_params = prior_carbs._basic_space_to_param_space(
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

            logger.info(
                f"Loaded {len(self.failure_observations)} failed observations from prior run"
            )

            self.resample_count = prior_carbs.resample_count

        if added_parameters is not None:
            for param_name, param_value in added_parameters.items():
                if param_name in self._real_number_space_by_name:
                    param_idx = ordered_dict_index(
                        self._real_number_space_by_name, param_name
                    )
                    param_value_in_basic = self._real_number_space_by_name[
                        param_name
                    ].basic_from_param(param_value)
                    self.search_center_in_basic[param_idx] = param_value_in_basic
