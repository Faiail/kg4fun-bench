# -*- coding: utf-8 -*-
from .ontomap_run import OntomapRun
from .run import Run
from .utils import ParameterKeys


def run_experiment(parameters: dict, cls: str, seed: int) -> None:
    parameters[ParameterKeys.SEED] = seed
    run_cls = globals()[cls]
    run_cls(parameters).launch()


__all__ = ["Run", "OntomapRun", "ParameterKeys", "run_experiment"]
