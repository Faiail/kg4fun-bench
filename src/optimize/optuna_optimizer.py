from src.experiment.utils import ParameterKeys
from .optimizer import HPOptimizer
from src.utils import load_ruamel
from src.utils.general import save_json
from copy import deepcopy
import src.experiment as experiment_pkg
import optuna
import os


def set_nested_value(d: dict, path: str, value) -> None:
    """Sets a value in a nested dictionary using a dot-separated path."""
    keys = path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


class OptunaOptimizer(HPOptimizer):
    def init(self):
        print("Init general...")
        self._init_general()
        print("Done!")
        print("Init Optuna Study...")
        self._init_study()
        print("Done!")
        print("Init Run...")
        self._init_run()
        print("Done!")

    def _init_general(self) -> None:
        general_parameters = self.parameters.get(ParameterKeys.GENERAL)
        self.out_dir = general_parameters.get(ParameterKeys.OUT_DIR, "./")
        os.makedirs(self.out_dir, exist_ok=True)
        self.seed = general_parameters.get(ParameterKeys.SEED, 42)

    def _init_study(self) -> None:
        study_parameters = self.parameters.get(ParameterKeys.STUDY)
        self.study_name = study_parameters.get(ParameterKeys.NAME)
        storage = f"sqlite:///{self.study_name}.db"
        study_config = study_parameters.get(ParameterKeys.CFG, dict())
        self.n_trials = study_parameters.get(ParameterKeys.N_TRIALS)
        self.space = study_parameters.get(ParameterKeys.SPACE)
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=storage,
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            **study_config,
        )

    def _init_run(self) -> None:
        run_parameters = self.parameters.get(ParameterKeys.RUN)
        self.run_name = run_parameters.get(ParameterKeys.NAME)
        self.run_config_fname = run_parameters.get(ParameterKeys.CFG)

    def sample_param(self, trial: optuna.Trial, name: str, spec: dict):
        p_type = spec.get("type")
        if p_type == "categorical":
            return trial.suggest_categorical(name, spec["choices"])
        elif p_type == "float":
            return trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=spec.get("log", False),
                step=spec.get("step", None),
            )
        elif p_type == "int":
            return trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                log=spec.get("log", False),
                step=spec.get("step", 1),
            )
        else:
            raise ValueError(f"Unsupported parameter type '{p_type}' for '{name}'.")

    def objective(self, trial: optuna.Trial) -> float:
        run_parameters = load_ruamel(deepcopy(self.run_config_fname))
        for param_name, param_spec in self.space.items():
            sampled_value = self.sample_param(trial, param_name, param_spec)
            target_path = param_spec.get("path", param_name)
            set_nested_value(run_parameters, target_path, sampled_value)
            run_parameters[ParameterKeys.GENERAL][
                ParameterKeys.OUT_DIR
            ] = f"{self.out_dir}/trial_{trial.number}"
            run_parameters[ParameterKeys.EARLY_STOP][ParameterKeys.CFG]["patience"] = 5

        try:
            runner = experiment_pkg.__dict__[self.run_name](run_parameters)
            runner.launch()
            best_score = runner.early_stop.best_score
            return float(best_score)
        except:
            return 0.0

    def optimize(self) -> dict:
        self.study.optimize(self.objective, n_trials=self.n_trials)
        results = {
            "best_value": self.study.best_value,
            "best_params": self.study.best_params,
        }
        save_json(results, f"{self.out_dir}/best.json")
