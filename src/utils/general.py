from ruamel.yaml import YAML
from pathlib import Path
import json


def load_json(file_path: str) -> dict:
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def save_json(data: dict, file_path: str):
    with open(file_path, "w+") as f:
        json.dump(data, f, indent=4)


def load_ruamel(path: str, typ: str = "safe") -> dict:
    yaml = YAML(typ=typ)
    return yaml.load(Path(path))
