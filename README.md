# Cost Aware pareto-Region Bayesian Search

CARBS depends primarily on pytorch and pyro for the Gaussian Process model. To get started, clone this directory and run,

```bash
pip install -e /path/to/carbs
```

## Using CARBS

The primary CARBS interface is through `suggest` (which will return a new point to test) and `observe` (to report the result).

See `notebooks/carbs_demo.sync.ipynb` for an example of how to use the optimizer.

Options for CARBS are described on the `CARBSParams` class in `carbs/utils.py`.

## Concepts

Here are some concepts to be familiar with when using CARBS:

### Cost

We usually use number of seconds of runtime as cost. 

It is recommended to start out the search in a low cost region, so the algorithm can get many iterations in quickly. If increasing the cost will increase the performance (as it usually does), CARBS will explore the higher cost area later. 

The `max_suggestion_cost` argument to `CARBSParams` is roughly used to cap the cost of suggestions. CARBS will not make any suggestions that it thinks will cost more than `max_suggestion_cost`. Because its cost model is not completely accurate, some suggestions will take longer than this time. They will not be truncated at the `max_suggestion_cost` amount of runtime.

### Success / Failure

CARBS keeps a separate model for whether a run will succeed or fail. Usually, we report a success if we are able to measure the target metric during eval at the end of training. A run should be reported as a failure if the hyperparameters suggested by CARBS caused the failure, for example a batch size that is too large that caused an OOM failure. If a failure occurs that is not related to the hyperparameters, it is better to forget the suggestion or retry it.

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
