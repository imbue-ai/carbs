import math
import os
import typing
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Sized
from typing import Tuple
from typing import Type
from typing import Union
from typing import cast

import attr
import numpy as np
import seaborn as sns
import torch
import wandb
from carbs.serialization import Serializable
from matplotlib import pyplot as plt
from scipy.special import wofz
from torch import Tensor
from torch.distributions import Normal

ParamType = Union[int, float, str, bool, Enum]
ParamDictType = Dict[str, ParamType]


@attr.s(auto_attribs=True, hash=True)
class ParamSpace(Serializable):
    def basic_from_param(self, value: ParamType) -> Any:
        raise NotImplementedError()

    def param_from_basic(self, value: Any) -> ParamType:
        raise NotImplementedError()

    def drop_type(self) -> Any:
        return self


@attr.s(auto_attribs=True, hash=True)
class RealNumberSpace(ParamSpace):
    min: float = float("-inf")
    max: float = float("+inf")
    scale: float = 1.0
    is_integer: bool = False
    rounding_factor: int = 1

    def basic_from_param(self, value: ParamType) -> float:
        raise NotImplementedError()

    def param_from_basic(self, value: float, is_rounded: bool = True) -> ParamType:
        raise NotImplementedError()

    def round_tensor_in_basic(self, value: Tensor) -> Tensor:
        if not self.is_integer:
            return value
        raise NotImplementedError()

    @property
    def min_bound(self):
        # compensate for floats being bad bounds for integers
        if self.is_integer:
            return self.min - 0.1
        return self.min

    @property
    def max_bound(self):
        # compensate for floats being bad bounds for integers
        if self.is_integer:
            return self.max + 0.1
        return self.max

    @property
    def plot_scale(self) -> str:
        raise NotImplementedError()


@attr.s(auto_attribs=True, hash=True)
class LinearSpace(RealNumberSpace):
    min: float = float("-inf")
    max: float = float("+inf")

    def basic_from_param(self, value: ParamType) -> float:
        assert isinstance(value, (int, float))
        return value / self.scale

    def param_from_basic(self, value: float, is_rounded: bool = True) -> float:
        value = value * self.scale
        if self.is_integer and is_rounded:
            value = round(value / self.rounding_factor) * self.rounding_factor
        return value

    def round_tensor_in_basic(self, value: Tensor) -> Tensor:
        if self.is_integer:
            return torch.round(value * self.scale / self.rounding_factor) * self.rounding_factor / self.scale
        else:
            return value

    @property
    def plot_scale(self) -> str:
        return "linear"


@attr.s(auto_attribs=True, hash=True)
class LogSpace(RealNumberSpace):
    min: float = 0.0
    max: float = float("+inf")
    base: int = 10

    def basic_from_param(self, value: ParamType) -> float:
        assert isinstance(value, (int, float))
        if value == 0.0:
            return float("-inf")
        return math.log(value, self.base) / self.scale

    def param_from_basic(self, value: float, is_rounded: bool = True) -> float:
        value = self.base ** (value * self.scale)
        if self.is_integer and is_rounded:
            value = round(value / self.rounding_factor) * self.rounding_factor
        return value

    def round_tensor_in_basic(self, value: Tensor) -> Tensor:
        if self.is_integer:
            rounded_value = (
                torch.round(self.base ** (value * self.scale) / self.rounding_factor) * self.rounding_factor
            )
            if self.base == 10:
                return torch.log10(rounded_value) / self.scale
            else:
                return torch.log(rounded_value) / self.scale / math.log(self.base)
        else:
            return value

    @property
    def plot_scale(self) -> str:
        return "log"


@attr.s(auto_attribs=True, hash=True)
class LogitSpace(RealNumberSpace):
    min: float = 0.0
    max: float = 1.0

    def basic_from_param(self, value: ParamType) -> float:
        assert isinstance(value, (int, float))
        if value == 0.0:
            return float("-inf")
        if value == 1.0:
            return float("+inf")
        return math.log10(value / (1 - value)) / self.scale

    def param_from_basic(self, value: float, is_rounded: bool = True) -> float:
        value = 1 / (10 ** (-value * self.scale) + 1)
        return value

    @property
    def plot_scale(self) -> str:
        return "logit"


@attr.s(auto_attribs=True)
class CategoricalSpace(ParamSpace):
    category_values: Tuple[ParamType, ...]
    category_probabilities: Tuple[float, ...]

    def __attrs_post_init__(self) -> None:
        assert len(self.category_values) == len(
            self.category_probabilities
        ), "category_names and category_probabilities must be equal length"
        assert len(self.category_values) > 0, "Zero choices is invalid for CategoricalSpace"

    def basic_from_param(self, value: ParamType) -> int:
        assert value in self.category_values
        return self.category_values.index(value)

    def param_from_basic(self, value: int) -> ParamType:
        return self.category_values[value]


@attr.s(auto_attribs=True)
class ConstantSpace(CategoricalSpace):
    def __attrs_post_init__(self) -> None:
        super(ConstantSpace, self).__attrs_post_init__()
        assert len(self.category_values) == 1


@attr.s(auto_attribs=True)
class BooleanSpace(CategoricalSpace):
    category_values: Tuple[bool, bool] = (False, True)
    category_probabilities: Tuple[float, float] = (0.5, 0.5)

    def basic_from_param(self, value: ParamType) -> int:
        assert value in self.category_values
        return self.category_values.index(value)

    def param_from_basic(self, value: int) -> bool:
        return self.category_values[value]


def from_bayesmark(bayesmark_config: Dict[str, Dict]) -> typing.OrderedDict[str, ParamSpace]:
    bones_config: typing.OrderedDict[str, ParamSpace] = OrderedDict()
    for name, param in bayesmark_config.items():
        if param["type"] in {"real", "int"}:
            # RealNumberSpace
            is_integer: bool
            if param["type"] == "real":
                is_integer = False
            elif param["type"] == "int":
                is_integer = True
                # make range a little bigger so we don't exclude the endpoints (should be inclusive)
                # This is now done later...
                # param["range"] = (param["range"][0] - 0.1, param["range"][1] + 0.1)
            else:
                raise NotImplementedError()
            space_type: Type[RealNumberSpace]
            if param["space"] == "linear":
                space_type = LinearSpace
            elif param["space"] == "log":
                space_type = LogSpace
            elif param["space"] == "logit":
                space_type = LogitSpace
            else:
                raise NotImplementedError

            space = space_type(min=param["range"][0], max=param["range"][1], is_integer=is_integer)
            with space.mutable_clone() as scaled_space:
                scaled_space.scale = space.basic_from_param(param["range"][1]) - space.basic_from_param(
                    param["range"][0]
                )
            bones_config[name] = scaled_space
        elif param["type"] == "bool":
            bones_config[name] = BooleanSpace()
        elif param["type"] == "cat":
            num_values = len(param["values"])
            probabilities = tuple([1 / num_values] * num_values)
            bones_config[name] = CategoricalSpace(
                category_values=param["values"], category_probabilities=probabilities
            )
    return bones_config


CategoricalTuple = Tuple[int, ...]
SUGGESTION_ID_DICT_KEY = "suggestion_uuid"


def log_norm_cdf(z: Tensor):
    """
    @MISC {256009,
        TITLE = {Approximation of logarithm of standard normal CDF for x&lt;0},
        AUTHOR = {Isaac Asher (https://stats.stackexchange.com/users/145180/isaac-asher)},
        HOWPUBLISHED = {Cross Validated},
        URL = {https://stats.stackexchange.com/q/256009}
    }
    This contains -z**2/2 which exactly cancels the same term in prior.log_prob for EI... We can then take the log
    out of the exp for the simplified form below
    """
    return torch.log(wofz(-z * 1j / math.sqrt(2)).real) - z**2 / 2 + math.log(0.5)


def expected_improvement(mu: Tensor, variance: Tensor, best_mu: Tensor, exploration_bias: float = 0.5) -> Tensor:
    prior = Normal(0, 1)
    sigma = variance.sqrt()
    z = (mu - best_mu - exploration_bias) / sigma
    # original form:
    # ei: Tensor = sigma * torch.exp(prior.log_prob(z)) * (1 + z * torch.exp(log_norm_cdf(z) - prior.log_prob(z)))
    # simplified form:
    wofz_output = wofz(-z.cpu() * 1j / math.sqrt(2)).real.to(z.device)
    ei: Tensor = sigma * torch.exp(prior.log_prob(z)) * (1 + z * wofz_output * math.sqrt(math.pi / 2))
    return ei


def probability_of_improvement(
    mu: Tensor, variance: Tensor, best_mu: Tensor, better_direction_sign: int, exploration_bias: float = 0.0
) -> Tensor:
    prior = Normal(0, 1)
    mu_improvement = (mu - best_mu) * better_direction_sign - exploration_bias
    sigma = variance.sqrt()
    poi: Tensor = prior.cdf(mu_improvement / sigma)
    return poi


def aggregate_logical_and_across_dim(x: Tensor, dim: int = -1) -> Tensor:
    """
    Takes in boolean x, aggregates across dimension dim
    """
    return cast(Tensor, torch.min(torch.where(x, 1, 0), dim=dim).values > 0)


def add_dict_key_prefix(input_dict: Dict[str, Any], prefix: str):
    return {f"{prefix}{k}": v for k, v in input_dict.items()}


@attr.s(auto_attribs=True, collect_by_mro=True)
class ObservationInParam(Serializable):
    input: ParamDictType
    output: float
    cost: float = 1.0
    is_failure: bool = False


@attr.s(auto_attribs=True, collect_by_mro=True)
class ObservationInBasic(Serializable):
    real_number_input: Tensor
    output: float
    cost: float = 1.0
    suggestion_id: Optional[str] = None


@attr.s(auto_attribs=True, collect_by_mro=True)
class SuggestionInBasic(Serializable):
    real_number_input: Tensor
    log_info: Dict[str, float] = attr.Factory(dict)


@attr.s(auto_attribs=True, collect_by_mro=True)
class ObservationInSurrogate(Serializable):
    input_in_natural: Tensor
    input_in_basic: Tensor
    value_in_surrogate: Tensor
    variance: Tensor
    probability: Tensor
    information_gain: Tensor
    distance_to_nearest_observation: Tensor


class AcquisitionFunctionEnum(Enum):
    EXPECTED_IMPROVEMENT = "EXPECTED_IMPROVEMENT"
    PROBABILITY_OF_IMPROVEMENT = "PROBABILITY_OF_IMPROVEMENT"


class SearchSpaceAcquisitionBiasEnum(Enum):
    PROBABILITY = "PROBABILITY"
    INFORMATION_GAIN = "INFORMATION_GAIN"
    NONE = "NONE"


class SurrogateFunctionEnum(Enum):
    STANDARDIZE = "STANDARDIZE"
    STANDARDIZE_LOG_NEG = "STANDARDIZE_LOG_NEG"


class UtilityFunctionEnum(Enum):
    IDENTITY = "IDENTITY"
    SORTED_INDEX = "SORTED_INDEX"
    CONFIDENCE_BOUND = "CONFIDENCE_BOUND"


class SearchDistributionFunctionEnum(Enum):
    NORMAL = "NORMAL"
    CAUCHY = "CAUCHY"


class QuasiRandomSamplerEnum(Enum):
    RANDOM = "RANDOM"
    HALTON = "HALTON"
    SCR_HALTON = "SCR_HALTON"
    SOBOL = "SOBOL"
    SCR_SOBOL = "SCR_SOBOL"


class OutstandingSuggestionEstimatorEnum(Enum):
    MEAN = "MEAN"
    CONSTANT = "CONSTANT"
    THOMPSON = "THOMPSON"


class SuggestionRedistributionMethodEnum(Enum):
    NONE = "NONE"
    LOG_COST_CLAMPING = "LOG_COST_CLAMPING"


@attr.s(auto_attribs=True, collect_by_mro=True)
class WandbLoggingParams(Serializable):
    project_name: Optional[str] = None
    group_name: Optional[str] = None
    run_name: Optional[str] = None
    run_id: Optional[str] = None
    is_suggestion_logged: bool = True
    is_observation_logged: bool = True
    is_search_space_logged: bool = True
    root_dir: str = "/mnt/private"


@attr.s(auto_attribs=True, collect_by_mro=True)
class CARBSParams(Serializable):
    better_direction_sign: int = 1  # 1 for maximizing, -1 for minimizing
    seed: int = 0
    num_random_samples: int = 4  # will do random suggestions until this many observations are made
    is_wandb_logging_enabled: bool = True
    wandb_params: WandbLoggingParams = WandbLoggingParams()
    is_saved_on_every_observation: bool = True

    initial_search_radius: float = 0.3  # Search radius in BASIC space

    exploration_bias: float = 1.0  # hyperparameter biasing BO acquisition function toward exploration

    num_candidates_for_suggestion_per_dim: int = 100

    resample_frequency: int = 5  # resample a pareto point every n observations, set 0 to disable

    max_cost: Optional[float] = None  # Will not make suggestions with predicted cost above this value

    min_pareto_cost_fraction: float = 0.2  # takes minimum cost for pareto set to be this percentile of cost data
    is_pareto_group_selection_conservative: bool = True
    is_expected_improvement_pareto_value_clamped: bool = True
    is_expected_improvement_value_always_max: bool = False

    outstanding_suggestion_estimator: OutstandingSuggestionEstimatorEnum = OutstandingSuggestionEstimatorEnum.THOMPSON


@attr.s(auto_attribs=True, collect_by_mro=True)
class SurrogateModelParams(Serializable):
    real_dims: int
    better_direction_sign: int  # 1 for maximizing, -1 for minimizing
    device: str = "cpu"
    min_category_observations: int = 3
    scale_length: float = 1
    outstanding_suggestion_estimator: OutstandingSuggestionEstimatorEnum = OutstandingSuggestionEstimatorEnum.MEAN


@attr.s(auto_attribs=True, collect_by_mro=True)
class SuggestOutput:
    suggestion: ParamDictType
    log: Dict[str, Any] = attr.Factory(dict)


@attr.s(auto_attribs=True, collect_by_mro=True)
class ObserveOutput:
    logs: Dict[str, Any] = attr.Factory(dict)


def load_observations_from_wandb_run(
    run_name: str, prefix: str = "observation/", add_params: Optional[ParamDictType] = None
) -> List[ObservationInParam]:
    api = wandb.Api()
    run = api.run(run_name)
    history_df = run.history()
    observations: List[ObservationInParam] = []
    for idx, row in history_df[[x for x in history_df.keys() if x.startswith(prefix)]].dropna().iterrows():
        input: Dict[str, ParamType] = {}
        if add_params is not None:
            input.update(add_params)
        output = float("inf")
        for k, v in row.to_dict().items():
            if k == f"{prefix}output":
                output = v
            else:
                input[k[len(prefix) :]] = v
        observations.append(ObservationInParam(input=input, output=output))

    return observations


def load_latest_checkpoint_from_wandb_run(run_path: str, temp_dir: Optional[str] = None) -> str:
    api = wandb.Api()
    run = api.run(run_path)
    # TODO: make this not hard coded?
    checkpoint_filenames = [
        file.name for file in run.files() if file.name.startswith("bones_") and file.name.endswith("obs.pt")
    ]
    checkpoint_filenames = sorted(checkpoint_filenames, key=lambda x: int(x[6:-6]))
    latest_checkpoint_filename = checkpoint_filenames[-1]
    return load_checkpoint_from_wandb_run(run_path, latest_checkpoint_filename, temp_dir)


def load_checkpoint_from_wandb_run(run_path: str, checkpoint_filename: str, temp_dir: Optional[str] = None) -> str:
    if temp_dir is None:
        temp_dir = f"/tmp/bones/{run_path}"
        os.makedirs(temp_dir, exist_ok=True)

    checkpoint_path = wandb.restore(checkpoint_filename, run_path=run_path, replace=True, root=temp_dir)
    assert checkpoint_path is not None, "Could not load checkpoint"
    return checkpoint_path.name


def assert_empty(x: Sized, message: str = "unexpected elements") -> None:
    assert len(x) == 0, f"{message}: {x}"


def ordered_dict_index(od: OrderedDict, value: Any) -> int:
    for i, k in enumerate(od.keys()):
        if k == value:
            return i
    raise KeyError(f"{value} not found in {od}")


def group_observations(observations_in_basic: List[ObservationInBasic]) -> List[Tuple[ObservationInBasic, ...]]:
    """
    Gets observations grouped by matching input params
    """
    outputs: List[Tuple[ObservationInBasic, ...]] = []
    observations = observations_in_basic.copy()
    observations.sort(key=lambda x: tuple(v.item() for v in x.real_number_input))
    while len(observations) > 0:
        obs = observations.pop()
        nearby_obs = [obs]
        for i in range(len(observations) - 1, -1, -1):
            if torch.all(torch.isclose(observations[i].real_number_input, obs.real_number_input)):
                nearby_obs.append(observations.pop(i))
        outputs.append(tuple(nearby_obs))

    return outputs


def observation_group_cost(group: Sequence[ObservationInBasic]) -> float:
    return sum(obs.cost for obs in group) / len(group)


def observation_group_output(group: Sequence[ObservationInBasic]) -> float:
    return sum(obs.output for obs in group) / len(group)


def pareto_area_from_groups(obs_groups: List[Tuple[ObservationInBasic, ...]]) -> float:
    if len(obs_groups) < 2:
        return 0
    last_cost = observation_group_cost(obs_groups[0])
    last_output = observation_group_output(obs_groups[0])
    total_area = 0.0
    for group in obs_groups[1:]:
        new_cost = observation_group_cost(group)
        new_output = observation_group_output(group)
        total_area += (new_cost - last_cost) * last_output
        last_cost = new_cost
        last_output = new_output
    return total_area


def get_pareto_groups(
    grouped_observations: List[Tuple[ObservationInBasic, ...]],
    min_pareto_cost_fraction: float,
    better_direction_sign: int,
) -> List[Tuple[ObservationInBasic, ...]]:
    min_pareto_cost = np.quantile(
        np.array([x.cost for group in grouped_observations for x in group]), min_pareto_cost_fraction
    )
    observations_below_min_threshold = [
        group for group in grouped_observations if observation_group_cost(group) <= min_pareto_cost
    ]

    group_output_pos_better = lambda x: observation_group_output(x) * better_direction_sign
    first_pareto_group = max(observations_below_min_threshold, key=group_output_pos_better)

    remaining_observations = [
        group for group in grouped_observations if observation_group_cost(group) > min_pareto_cost
    ]
    remaining_observations.sort(key=observation_group_cost)

    pareto_groups: List[Tuple[ObservationInBasic, ...]] = [first_pareto_group]
    best_output = group_output_pos_better(first_pareto_group)
    for obs_group in remaining_observations:
        mean_output = group_output_pos_better(obs_group)
        if mean_output > best_output:
            pareto_groups.append(obs_group)
            best_output = mean_output

    return pareto_groups


def get_pareto_groups_conservative(
    grouped_observations: List[Tuple[ObservationInBasic, ...]],
    min_pareto_cost_fraction: float,
    better_direction_sign: int,
) -> List[Tuple[ObservationInBasic, ...]]:
    """
    Just like get_pareto_groups but prefers groups with multiple samples. A single sample can only be "better" if
    it is better than the max of the previous group
    """

    min_pareto_cost = np.quantile(
        np.array([x.cost for group in grouped_observations for x in group]), min_pareto_cost_fraction
    )
    observations_below_min_threshold = [
        group for group in grouped_observations if observation_group_cost(group) <= min_pareto_cost
    ]
    resampled_observations_below_min_threshold = [
        group for group in observations_below_min_threshold if len(group) > 1
    ]

    # Only use resampled observations if there are any
    if len(resampled_observations_below_min_threshold) > 0:
        observations_below_min_threshold = resampled_observations_below_min_threshold

    group_output_pos_better = lambda x: observation_group_output(x) * better_direction_sign
    max_group_output_pos_better = lambda x: max([obs.output * better_direction_sign for obs in x])
    first_pareto_group = max(observations_below_min_threshold, key=group_output_pos_better)

    remaining_observations = [
        group for group in grouped_observations if observation_group_cost(group) > min_pareto_cost
    ]
    remaining_observations.sort(key=observation_group_cost)

    pareto_groups: List[Tuple[ObservationInBasic, ...]] = [first_pareto_group]
    best_output = group_output_pos_better(first_pareto_group)
    best_output_max = max_group_output_pos_better(first_pareto_group)
    for obs_group in remaining_observations:
        mean_output = group_output_pos_better(obs_group)
        # If only one sample, it must be better than the max of the last group
        # Otherwise, only must have better mean
        if (len(obs_group) > 1 and mean_output > best_output) or (
            len(obs_group) == 1 and mean_output > best_output_max
        ):
            pareto_groups.append(obs_group)
            best_output = mean_output
            best_output_max = max_group_output_pos_better(obs_group)

    return pareto_groups


def get_pareto_curve_plot(
    observations: List[ObservationInBasic],
    pareto_groups: List[Tuple[ObservationInBasic, ...]],
    save_dir: Optional[str] = None,
    obs_count: Optional[int] = None,
) -> Optional[str]:
    sns.set_theme(style="whitegrid")
    plt.clf()
    pareto_set = [obs for group in pareto_groups for obs in group]
    plt.scatter(
        [x.cost for x in pareto_set],
        [x.output for x in pareto_set],
        c="black",
        s=100,
        # marker="o",
    )
    plt.plot(
        [observation_group_cost(g) for g in pareto_groups],
        [observation_group_output(g) for g in pareto_groups],
        linewidth=4,
        color="black",
    )
    plt.scatter(
        [x.cost for x in observations],
        [x.output for x in observations],
        c=list(range(len(observations))),
        cmap="plasma",
    )
    plt.xscale("log")
    if obs_count is None:
        obs_count = len(observations)
    plt.title(f"Pareto curve at {obs_count} obs")
    plt.ylabel("performance")
    plt.xlabel("cost")

    if save_dir is not None:
        img_path = Path(save_dir) / f"pareto_curve_{obs_count}.png"
        plt.savefig(img_path)
        return str(img_path)
    return None
