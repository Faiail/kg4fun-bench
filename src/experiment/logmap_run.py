# -*- coding: utf-8 -*-
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List
import time

from src.experiment.run import Run
from src.experiment.utils import ParameterKeys
from src.models.ontomap.evaluation import evaluator_module
from src.utils.general import load_json, save_json

class LogmapRun(Run):
    def _init_general(self) -> None:
        general_params = self.parameters.get(ParameterKeys.GENERAL, {})
        self.pbar = general_params.get(ParameterKeys.PBAR, True)
        self.root_dir = general_params.get(ParameterKeys.ROOT_DIR, "data/processed")
        base_dir = Path(general_params.get(ParameterKeys.OUT_DIR, "experiments/outputs"))
        config_name = self.parameters.get(ParameterKeys.CONFIG_NAME, "logmap")
        seed = self.parameters.get(ParameterKeys.SEED, 42)
        self.out_dir = base_dir / f"{config_name}_{seed}"
        os.makedirs(self.out_dir, exist_ok=True)
        self.approach = "logmap"
        self.jar_path = os.path.abspath("apis/logmap/logmap-matcher.jar")

    def _init_model(self) -> None:
        self.model_name = "LogMap"

    def _init_encoder(self) -> None:
        self.encoder_id = "none"

    def _init_datasets(self) -> None:
        data_params = self.parameters.get(ParameterKeys.DATA, {})
        self.tracks = data_params.get(ParameterKeys.TRACKS, [])
        self.tasks = []
        for track in self.tracks:
            track_name = track.get(ParameterKeys.NAME)
            tasks = track.get(ParameterKeys.TASKS, [])
            for task in tasks:
                self.tasks.append(f"{track_name}/{task}")

    def _init_evaluation(self) -> None:
        eval_params = self.parameters.get(ParameterKeys.EVALUATION, {})
        self.do_evaluation = eval_params.get(ParameterKeys.DO_EVALUATION, True)
        self.llm_confidence_th = eval_params.get(ParameterKeys.LLM_CONFIDENCE_TH, 0.5)

    def _parse_logmap_alignments(self, xml_path: str) -> List[Dict]:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        ns = {
            'align': 'http://knowledgeweb.semanticweb.org/heterogeneity/alignment',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
        }
        
        predictions = []
        for cell in root.findall('.//align:Cell', ns):
            e1_node = cell.find('align:entity1', ns)
            e2_node = cell.find('align:entity2', ns)
            
            if e1_node is None or e2_node is None:
                continue
                
            e1 = e1_node.attrib.get(f'{{{ns["rdf"]}}}resource')
            e2 = e2_node.attrib.get(f'{{{ns["rdf"]}}}resource')
            
            if not e1 or not e2:
                continue
                
            measure_node = cell.find('align:measure', ns)
            measure = float(measure_node.text) if measure_node is not None else 1.0
            
            src_id = e1.split('#')[-1]
            tgt_id = e2.split('#')[-1]
            
            predictions.append({
                "source": src_id,
                "target": tgt_id,
                "score": measure
            })
        return predictions

    def launch(self) -> Any:
        print("\nStarting LogMap Experiment Run...")
        print(f"Output directory: {self.out_dir}")
        print(f"Total dataset tasks: {len(self.tasks)}\n")

        summary_metrics = {}

        for task in self.tasks:
            print("=" * 60)
            print(f"Working on task: {task}")
            print("=" * 60)
            
            melt_dir = os.path.join("data/processed/melt", task)
            source_rdf = os.path.join(melt_dir, "source.rdf")
            target_rdf = os.path.join(melt_dir, "target.rdf")
            
            if not os.path.exists(source_rdf) or not os.path.exists(target_rdf):
                print(f"Skipping {task}: MELT RDF files not found in {melt_dir}")
                continue
                
            task_name = task.split('/')[-1]
            task_out_dir = self.out_dir / task.split('/')[0] / task_name
            os.makedirs(task_out_dir, exist_ok=True)
            
            start_time = time.time()
            
            print("\tRunning LogMap Java executable...")
            subprocess.run([
                "java", "-jar", self.jar_path,
                "MATCHER",
                f"file://{os.path.abspath(source_rdf)}",
                f"file://{os.path.abspath(target_rdf)}",
                os.path.abspath(task_out_dir),
                "false"
            ], check=False, capture_output=True)
            
            elapsed = time.time() - start_time
            
            align_file = os.path.join(task_out_dir, "logmap2_mappings.rdf")
            if not os.path.exists(align_file):
                print(f"\tError: LogMap failed to produce {align_file}")
                continue
                
            print("\tParsing LogMap predictions...")
            predicts = self._parse_logmap_alignments(align_file)
            print(f"\tPredictions generated: {len(predicts)} in {elapsed:.2f}s")
            
            save_json(predicts, os.path.join(task_out_dir, "logmap_predictions.json"))
            
            if self.do_evaluation:
                print("\tEvaluating predictions...")
                ref_path = os.path.join("data/processed", task, "alignment/alignment.json")
                references_raw = load_json(ref_path)
                
                eval_results = evaluator_module(
                    track=task.split('/')[0],
                    approach=self.approach,
                    predicts=predicts,
                    references=references_raw,
                    llm_confidence_th=self.llm_confidence_th,
                )
                print(f"\tEvaluation results: {eval_results}")
                
                summary_metrics[task] = eval_results

        if self.do_evaluation:
            save_json(summary_metrics, os.path.join(self.out_dir, "summary_metrics.json"))
            print(f"\nExperiment complete. Summary metrics saved in {os.path.join(self.out_dir, 'summary_metrics.json')}")

