# -*- coding: utf-8 -*-
import os
from pathlib import Path
import time
from typing import Any, Dict, List

from src.experiment.run import Run
from src.experiment.utils import ParameterKeys
from src.models.ontomap.encoder import EncoderCatalog
from src.models.ontomap.evaluation import evaluator_module
from src.models.ontomap.ontology import ontology_matching
from src.models.ontomap.ontology_matchers import MatcherCatalog
from src.utils.general import save_json


class OntomapRun(Run):
    def _init_general(self) -> None:
        general_params = self.parameters.get(ParameterKeys.GENERAL, {})
        self.device = general_params.get(ParameterKeys.DEVICE, ParameterKeys.CPU)
        self.pbar = general_params.get(ParameterKeys.PBAR, True)
        self.root_dir = general_params.get(ParameterKeys.ROOT_DIR, "data/processed")
        base_dir = Path(general_params.get(ParameterKeys.OUT_DIR, "experiments/outputs"))
        config_name = self.parameters.get(ParameterKeys.CONFIG_NAME, "ontomap")
        seed = self.parameters.get(ParameterKeys.SEED, 42)
        self.out_dir = base_dir / f"{config_name}_{seed}"
        os.makedirs(self.out_dir, exist_ok=True)
        self.approach = self.parameters.get(ParameterKeys.APPROACH, "lightweight")

    def _init_model(self) -> None:
        model_params = self.parameters.get(ParameterKeys.MODEL, {})
        self.model_name = model_params.get(ParameterKeys.NAME, "SimpleFuzzySM")
        self.model_kwargs = model_params.get(ParameterKeys.PARAMS, {})

        if self.approach not in MatcherCatalog:
            raise ValueError(
                f"Approach '{self.approach}' not found in MatcherCatalog. "
                f"Available approaches: {list(MatcherCatalog.keys())}"
            )
        if self.model_name not in MatcherCatalog[self.approach]:
            raise ValueError(
                f"Model '{self.model_name}' not found for approach '{self.approach}'. "
                f"Available models: {list(MatcherCatalog[self.approach].keys())}"
            )
        model_cls = MatcherCatalog[self.approach][self.model_name]
        self.model = model_cls(**self.model_kwargs)

    def _init_encoder(self) -> None:
        encoder_params = self.parameters.get(ParameterKeys.ENCODER, {})
        self.encoder_approach = encoder_params.get(
            ParameterKeys.APPROACH, self.approach
        )
        self.encoder_id = encoder_params.get(
            ParameterKeys.ENCODER_ID,
            encoder_params.get(ParameterKeys.NAME, "label"),
        )

        if self.encoder_approach not in EncoderCatalog:
            raise ValueError(
                f"Encoder approach '{self.encoder_approach}' not found in EncoderCatalog. "
                f"Available approaches: {list(EncoderCatalog.keys())}"
            )
        if self.encoder_id not in EncoderCatalog[self.encoder_approach]:
            raise ValueError(
                f"Encoder ID '{self.encoder_id}' not found for approach '{self.encoder_approach}'. "
                f"Available encoders: {list(EncoderCatalog[self.encoder_approach].keys())}"
            )
        encoder_cls = EncoderCatalog[self.encoder_approach][self.encoder_id]
        self.encoder = encoder_cls()

    def _init_datasets(self) -> None:
        dataset_params = self.parameters.get(ParameterKeys.DATASET, {})
        self.tracks = dataset_params.get(ParameterKeys.TRACKS, ["kg4fun"])
        self.tasks_to_consider = dataset_params.get(ParameterKeys.TASKS, None)
        self.load_from_json = dataset_params.get(
            ParameterKeys.LOAD_FROM_JSON, False
        )

        self.dataset_tasks: List[Any] = []
        for track in self.tracks:
            if track in ontology_matching:
                for task_cls in ontology_matching[track]:
                    task_obj = task_cls()
                    if (
                        self.tasks_to_consider is None
                        or task_obj.ontology_name in self.tasks_to_consider
                    ):
                        # Avoid duplicates if multiple tracks alias to the same dataset
                        if not any(
                            t.track == task_obj.track
                            and t.ontology_name == task_obj.ontology_name
                            for t in self.dataset_tasks
                        ):
                            self.dataset_tasks.append(task_obj)

    def _init_evaluation(self) -> None:
        eval_params = self.parameters.get(ParameterKeys.EVALUATION, {})
        self.do_evaluation = eval_params.get(ParameterKeys.DO_EVALUATION, True)
        self.llm_confidence_th = float(
            eval_params.get(ParameterKeys.LLM_CONFIDENCE_TH, 0.7)
        )

    def run_task(self, task_obj: Any) -> Dict[str, Any]:
        task_str = str(task_obj)
        print(f"\n{'='*60}")
        print(f"Working on task: {task_str}")
        print(f"{'='*60}")

        if self.load_from_json:
            task_owl = task_obj.load_from_json(root_dir=self.root_dir)
        else:
            task_owl = task_obj.collect(root_dir=self.root_dir)

        if self.approach == "rag":
            task_owl["llm"] = self.model_name

        print(f"\tEncoding inputs using '{self.encoder_id}' encoder...")
        encoded_inputs = self.encoder(**task_owl)

        print(f"\tGenerating predictions using '{self.model_name}'...")
        start_time = time.time()
        try:
            predicts = self.model.generate(input_data=encoded_inputs)
        except RuntimeError as e:
            print(f"\tRuntime error during generation: {e}")
            predicts = []
        elapsed = time.time() - start_time
        print(f"\tPredictions generated: {len(predicts)} in {elapsed:.2f}s")

        eval_results = None
        if self.do_evaluation:
            print("\tEvaluating predictions...")
            eval_results = evaluator_module(
                track=task_owl["dataset-info"]["track"],
                approach=self.approach,
                predicts=predicts,
                references=task_owl["reference"],
                llm_confidence_th=self.llm_confidence_th,
            )
            print(f"\tEvaluation results: {eval_results}")

        task_track = task_owl["dataset-info"]["track"]
        task_onto_name = task_owl["dataset-info"]["ontology-name"]
        task_out_dir = self.out_dir / task_track / task_onto_name
        os.makedirs(task_out_dir, exist_ok=True)

        output_dict = {
            "model": self.model_name,
            "approach": self.approach,
            "encoder-id": self.encoder_id,
            "dataset-info": task_owl["dataset-info"],
            "response-time": elapsed,
            "predictions-count": len(predicts),
            "evaluation-results": eval_results,
            "generated-output": predicts,
        }

        output_file_path = (
            task_out_dir
            / f"{self.approach}_{self.model_name}_{self.encoder_id}_output.json"
        )
        save_json(output_dict, str(output_file_path))
        print(f"\tSaved task output to {output_file_path}")

        return output_dict

    def launch(self) -> Dict[str, Any]:
        print("\nStarting OntoMap Experiment Run...")
        print(f"Output directory: {self.out_dir}")
        print(f"Total dataset tasks: {len(self.dataset_tasks)}")

        summary = {
            "config": self.parameters,
            "approach": self.approach,
            "model": self.model_name,
            "encoder": self.encoder_id,
            "tasks": {},
        }

        for task_obj in self.dataset_tasks:
            task_result = self.run_task(task_obj)
            task_key = f"{task_obj.track}/{task_obj.ontology_name}"
            summary["tasks"][task_key] = {
                "response-time": task_result["response-time"],
                "predictions-count": task_result["predictions-count"],
                "evaluation-results": task_result["evaluation-results"],
            }

        summary_file = self.out_dir / "summary_metrics.json"
        save_json(summary, str(summary_file))
        print(f"\nExperiment complete. Summary metrics saved in {summary_file}")
        return summary
