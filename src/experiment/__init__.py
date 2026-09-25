# -*- coding: utf-8 -*-
from .ontomap_run import OntomapRun
from .logmap_run import LogmapRun
from .run import Run
from .utils import ParameterKeys


def run_experiment(parameters: dict, cls: str, seed: int) -> None:
    parameters[ParameterKeys.SEED] = seed
    run_cls = globals()[cls]
    run_cls(parameters).launch()


__all__ = ["Run", "OntomapRun", "LogmapRun", "ParameterKeys", "run_experiment"]
