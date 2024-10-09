import os
import warnings
from functools import partial
from typing import Dict, Any

import optuna
from monai.bundle.config_parser import ConfigParser
from lighter.utils.runner import parse_config
from lighter.utils.dynamic_imports import import_module_from_path

def get_lighter_parser(config: Dict[str, Any]) -> ConfigParser:
    """
    Create and return a Lighter parser with the given configuration.

    Args:
        config (Dict[str, Any]): Configuration parameters for the parser.

    Returns:
        ConfigParser: Configured Lighter parser.
    """
    parser = parse_config(**config)
    project = parser.get_parsed_content("project")
    if project is not None:
        import_module_from_path("project", project)
    return parser

def objective(trial: optuna.trial.Trial, base_config: Dict[str, Any], hyperparam_dict: Dict[str, Dict[str, Any]], monitor: str) -> float:
    """
    Objective function for Optuna optimization.

    Args:
        trial (optuna.trial.Trial): Current trial object.
        base_config (Dict[str, Any]): Base configuration for the experiment.
        hyperparam_dict (Dict[str, Dict[str, Any]]): Hyperparameters to optimize.
        monitor (str): Metric to monitor for optimization.

    Returns:
        float: Value of the monitored metric for this trial.
    """
    # Suggest values for hyperparameters
    for key, value in hyperparam_dict.items():
        method = f"suggest_{value['type']}"
        if hasattr(trial, method):
            suggest_func = getattr(trial, method)
            if value['type'] == "categorical":
                trial.params[key] = suggest_func(key, value['choices'])
            else:
                trial.params[key] = suggest_func(key, *value['range'], log=value.get('log', False))
    
    # Update base config with trial params
    trial_config = base_config.copy()
    trial_config.update(trial.params)
    
    # Generate a unique name based on trial params
    param_str = "_".join([f"{k}_{v:.4f}" if isinstance(v, float) else f"{k}_{v}" for k, v in trial.params.items()])
    unique_name = f"{trial_config['vars#name']}_{param_str}".replace("#", "_")
    trial_config['vars#name'] = unique_name
    
    # Set up and run the experiment
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parser = get_lighter_parser(trial_config)
        system = parser.get_parsed_content("system")
        trainer = parser.get_parsed_content("trainer")

    trainer.fit(system)
    return trainer.callback_metrics[monitor].item()

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run hyperparameter optimization")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    args = parser.parse_args()

    # Load meta-configuration
    meta_config = ConfigParser().load_config_files(args.config)

    hyperparam_config = meta_config["hyperparam_config"]
    base_config = meta_config["base"]

    # Ensure required keys are present in hyperparam_config
    if "monitor" not in hyperparam_config:
        raise ValueError("'monitor' must be specified in hyperparam_config")
    
    monitor = hyperparam_config.pop("monitor")
    direction = hyperparam_config.pop("direction", "maximize")
    n_trials = hyperparam_config.pop("n_trials", 100)

    # Create and run Optuna study
    study = optuna.create_study(direction=direction)
    study.optimize(
        partial(objective, base_config=base_config, hyperparam_dict=hyperparam_config, monitor=monitor),
        n_trials=n_trials
    )

    print(f"Best trial: {study.best_trial.params}")
    print(f"Best {monitor}: {study.best_value}")

if __name__ == "__main__":
    main()
