from pathlib import Path

import click

from src.utils import load_ruamel


@click.group()
def main():
    pass

@main.command("preprocess")
@click.option("--parameters", help="Path to the parameters YAML file", required=True)
@click.option("--cls", help="Name of the preprocessing class to use", required=True)
def preprocess(parameters, cls):
    from src.preprocess import preprocess_fn
    preprocess_fn(load_ruamel(parameters), cls)


@main.command("experiment")
@click.option("--parameters", help="Path to the experiment parameters YAML file", required=True)
@click.option("--cls", help="Run class to use (e.g. OntomapRun)", default="OntomapRun", show_default=True)
@click.option("--seed", help="Random seed", default=42, type=int, show_default=True)
def experiment(parameters, cls, seed):
    from src.experiment import run_experiment

    params_dict = load_ruamel(parameters)
    params_dict["config_name"] = Path(parameters).stem
    run_experiment(params_dict, cls, seed)


@main.command("optimize")
@click.option("--parameters", help="Path to the experiment parameters YAML file", required=True)
@click.option("--cls", help="Run class to use", show_default=True)
def optimize(parameters, cls):
    from src.optimize import run_optimization

    params_dict = load_ruamel(parameters)
    params_dict["config_name"] = Path(parameters).stem
    run_optimization(params_dict, cls)


@main.command("profile")
@click.option("--parameters", help="Path to the experiment parameters YAML file", required=True)
@click.option("--cls", help="Run class to use (e.g. SharedCrossTextSingleItemRun)", required=True)
@click.option("--batch-size", help="Override batch size", default=None, type=int)
@click.option("--hidden-size", help="Override final hidden size", default=None, type=int)
@click.option("--steps", help="Number of training batches to profile (default: 10)", default=10, type=int)
@click.option("--full-epoch", help="Run a full epoch instead of fixed steps", is_flag=True, default=False)
def profile(parameters, cls, batch_size, hidden_size, steps, full_epoch):
    from src.experiment.profile_cross_memory import run_memory_profile

    params_dict = load_ruamel(parameters)
    params_dict["config_name"] = Path(parameters).stem
    run_memory_profile(
        parameters=params_dict,
        cls=cls,
        batch_size=batch_size,
        hidden_size=hidden_size,
        steps=steps,
        full_epoch=full_epoch,
    )

