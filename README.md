# Cost Aware pareto-Region Bayesian Search

CARBS depends primarily on pytorch and pyro for the Gaussian Process model. To get started, clone this directory and run,

```bash
pip install -e /path/to/carbs
```

## Using CARBS

The primary CARBS interface is through `suggest` (which will return a new point to test) and `observe` (to report the result).

See `notebooks/carbs_demo.sync.ipynb` for an example of how to use the optimizer.

Options for CARBS are described on the `CARBSParams` class in `carbs/utils.py`.



## Internal Usage Guide

This example is based on the `avalon/agent` project in the codebase as of Aug 2022 (commit 85a9d44cfe4dd4b4c7357a244fc16d1625192082 if you want to follow the exact example). It will probably be out of date by the time you're reading this, but the details are less important than conveying the general idea of how this tool is used.

Components:
- the `bones` code (in this folder). You shouldn't need to understand or modify this.
- your project (training code, etc). in this example, this is the code in `standalone/avalon/agent` (specifically the code used to train a PPO agent).
- a python file that interfaces bones and your project. in this example, this is the file `standalone/avalon/opt_ppo.py`. this file:
    - uses the Bones API to configure the hyperparameters to sweep over, to record the results of each run, to generate hyperparameter suggestions for each run in the optimization, and to push metrics to a wandb run specific to the bones optimizaiton. 
    - imports, configures and launches your project's code for each run in the optimization loop.
    - uses the computronium API to allocate compute resources to run each training run in the optimization loop.


Specifically, here's (loosely) what's happening inside a Bones run:
- you launch an "experiment" (in the `science` terminology) for the overall bones experiment (using `science`)
  - eg, from your local computer, run something like `python science/bin/science launch_experiment --project standalone/avalon --cluster_spec provider:physical,priority:4000,gpus:0 --name CARBS_RUN_NAME --command "opt_ppo.py bones run"`
    - this is a CPU-only experiment, as it's just running the bones logic and initiating the launch of the the individual experiments onto separate resources.
    - TBD: what priority to make this? doesn't really matter since i don't think we currently evict CPU containers.
  - this calls that `opt_ppo.py` file with the command `bones run`, which sets up an  `OptimizationExperiment` and calls its `run()` method - this is the bones logic.
- initialization (set up the bones wandb run, configure hyperparameters to optimize, etc)
- optimization loop (in practice, multiple runs will be happening in parallel):
  - bones generates a set of hyperparameters to use in this run (a "suggestion").
    - see [this notion doc](notion.so/generallyintelligent/Ellie-s-BONES-readme-d565d9a7d8084bc1a85354fbdb2e22a4) for an explainer on how Bones actually models the hyperparameter space and generates suggestions.
  - this gets compiled into a command to run on a new machine, which will run the actual experiment. This command will have a form like `python opt_ppo.py trainable train HYPERPARAM_FLAGS`. This initializes a `RegularTrainCommand` and calls its `train()` method, which is where you should implement the logic to initialize and launch your experiment.
  - launches a new container somewhere using `computronium`, and passes it the training command.
  - the experiment runs on the new container to completion
  - the result is gathered - the overall "score" metric is captured (out of the experiment log output, i think), and this score is passed to bones to update the optimizer state. if the run fails, you can configure it to either record a score of 0, or ignore that result.



