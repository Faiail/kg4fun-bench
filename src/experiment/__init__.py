# -*- coding: utf-8 -*-
from .ontomap_run import OntomapRun
from .logmap_run import LogmapRun
from .run import Run
from .utils import ParameterKeys
from .shared_text_single_item_run import SharedTextSingleItemRun
from .dedicated_text_single_item_run import DedicatedTextSingleItemRun
from .shared_text_complete_run import SharedTextCompleteRun
from .dedicated_text_complete_run import DedicatedTextCompleteRun


def run_experiment(parameters: dict, cls: str, seed: int) -> None:
    parameters[ParameterKeys.SEED] = seed
    run_cls = globals()[cls]
    run_cls(parameters).launch()


__all__ = ["Run", "OntomapRun", "LogmapRun", "ParameterKeys", "run_experiment"]
