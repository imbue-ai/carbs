# Cost Aware pareto-Region Bayesian Search

CARBS is a hyperparameter optimizer that can optimize both regular hyperparameters (like learning rate) and cost-related hyperparameters (like the number of epochs over data). It is a local search algorithm, so it benefits significantly from a good starting point. It searches around the pareto frontier of cost and performance, making it effective in finding compute efficient solutions to problems. See more in [our paper](https://arxiv.org/abs/2306.08055) or the [related blog post](https://imbue.com/research/carbs/). We have [used CARBS extensively](https://imbue.com/research/70b-carbs/) in training and scaling up large language models.

## Installing

CARBS depends primarily on pytorch and pyro for the Gaussian Process model. To get started, clone this directory and run,

```bash
pip install -e /path/to/carbs
```

## Using CARBS

The primary CARBS interface is through `suggest` (which will return a new point to test) and `observe` (to report the result).

Here is the core part of calling CARBS, (for the full example see `notebooks/carbs_demo.ipynb`):

```python
param_spaces = [
    Param(name="learning_rate", space=LogSpace(scale=0.5), search_center=1e-4),
    Param(name="momentum", space=LogitSpace(), search_center=0.9),
    Param(name="epochs", space=LogSpace(is_integer=True, min=2, max=512), search_center=10),
]
carbs_params = CARBSParams(
    better_direction_sign=-1,
    is_wandb_logging_enabled=False,
    resample_frequency=0,
)
carbs = CARBS(carbs_params, param_spaces)
for i in range(10):
    suggestion = carbs.suggest().suggestion
    observed_value = run_test_fn(suggestion)
    obs_out = carbs.observe(ObservationInParam(input=suggestion, output=observed_value, cost=suggestion["epochs"]))
```

By default, suggestions will be remembered and included in the GP model using Thompson sampling, to avoid suggesting the same point repeatedly if experiments are being done in parallel. Use `suggest(is_suggestion_remembered=False)` to disable this behavior.

### Configuration

Options for CARBS are described on the `CARBSParams` class in `carbs/utils.py`.

On the configuration class `CARBSParams`, be sure to set:
* `better_direction_sign` to 1 to find a maximum or -1 to find a minimum. 
* `wandb_params` there to configure wandb logging. 

Optionally also set:
* `max_suggestion_cost` (defaults to None) which is a soft restriction on the maximum cost of a suggestion made by the algorithm. This uses the GP model of the cost, so it may not be completely accurate early on in an experiment.
* `num_random_samples` (defaults to 4) will be the number of observations seen before CARBS starts make its own suggestions
* `is_saved_on_every_observation` (defaults to True) will pickle and save the whole class to the wandb run on each observation
* `resample_frequency` (defaults to 5) will be the frequency at which CARBS resamples points on the pareto front (starting from the lowest cost point). Set to 0 to disable resampling
* `min_pareto_cost_fraction` (defaults to 0.2) has the effect of bucketing together the lowest cost observations -- in the default case 20% -- As these observations are typically much noisier and less interesting than the high cost observations. Set to 0.0 to disable.
* `initial_search_radius` (defaults to 0.3) will change the scale over all search variables. We've found 0.3 to be fairly good across different types of problems.


### Search space

CARBS only supports continuous and integer search spaces. The spaces do not need to have bounds, but `min` and `max` values may be specified. The three main types are:
* `LinearSpace`: Be sure to set a `scale` parameter to describe a relevant scale length for the problem. If you are using an integer space and the default radius of 0.3, you will need to choose a scale >3 to ensure that neighboring integers can be reached.
* `LogSpace`: Good for cost or scale related variables, as well as other typically log distributed continuous variables. If using an integer space, you will need to set a minimum value.
* `LogitSpace`: This will only output values between 0 to 1, so cannot be used with integer values.


## Concepts

Here are some concepts to be familiar with when using CARBS:

### Cost

We usually use number of seconds of runtime as cost. 

It is recommended to start out the search in a low cost region, so the algorithm can get many iterations in quickly. If increasing the cost will increase the performance (as it usually does), CARBS will explore the higher cost area later. 

The `max_suggestion_cost` argument to `CARBSParams` is roughly used to cap the cost of suggestions. CARBS will not make any suggestions that it thinks will cost more than `max_suggestion_cost`. Because its cost model is not completely accurate, some suggestions will take longer than this time. They will not be truncated at the `max_suggestion_cost` amount of runtime.

### Success / Failure

CARBS keeps a separate model for whether a run will succeed or fail. Usually, we report a success if we are able to measure the target metric during eval at the end of training. A run should be reported as a failure if the hyperparameters suggested by CARBS caused the failure, for example a batch size that is too large that caused an OOM failure. If a failure occurs that is not related to the hyperparameters, it is better to forget the suggestion or retry it. Report a failure by making an `ObservationInParam` with `is_failure=True`

### Basic / Param Space

We map parameter spaces into a more natural search space internally to CARBS for modeling purposes. We call the raw parameter space, used for input and output, **Parameter Space**. We map that to a **Basic Space** using the parameter type, so a `LogSpace` will be transformed by the `log`/`exp` functions. We also use the `scale` factor in this transformation.

### Integer spaces and rounding

Log and Linear spaces can take the flag `is_integer=True` and a `rounding_factor` to round to a nearest value (eg, to round to the nearest multiple of 8). One potential gotcha here is that if the `search_radius` (which defaults to 0.3) does not reach the next integer value, CARBS will not be able to vary this parameter. Adding a `scale` factor here that is at least `1/search_radius` is necessary for `LinearSpace` to work properly. `LogSpace` is a little more complicated, but if search space starts too small (<4) or at a small multiple of the `rounding_factor`, the same issues can occur and a higher `scale` may be required.

### Observations, Suggestions, and Candidates

* **Observations** are the result of a full experiment.
* **Suggestions** are points that have an outstanding request to get results from a full experiment, but which have not yet been observed.
* **Candidates** are points that are under consideration to become suggestions.

### Surrogate model fitting

The `SurrogateModel` builds a surrogate model for the function you are testing. It in turn has four fit functions, for different inputs:

* **fit_observations**: Used to fit `success_observations` to produce the initial models of the target function's outputs and costs.
* **fit_suggestions**: Modifies the model of the target function output to include predictions for outstanding suggestions, using either Thompson sampling or a kriging believer.
* **fit_failures**: Pass both `success_observations` and `failure_observations` to create a model of the failure probability
* **fit_pareto_set**: Pass observations in the pareto set, to create a model of the pareto output value versus cost.
