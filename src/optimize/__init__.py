from .optuna_optimizer import OptunaOptimizer


def run_optimization(parameters: dict, cls: str) -> None:
    run_cls = globals()[cls]
    run_cls(parameters).optimize()