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