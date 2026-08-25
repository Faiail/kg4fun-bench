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