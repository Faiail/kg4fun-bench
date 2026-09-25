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
from torch_geometric.seed import seed_everything
import torch
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score


class LogmapRun(Run):
    def init(self) -> None:
        self._init_metrics()
        return super().init()

    def _init_general(self) -> None:
        general_parameters = self.parameters.get(ParameterKeys.GENERAL, dict())
        self.pbar = general_parameters.get(ParameterKeys.PBAR, False)
        out_dir = general_parameters.get(ParameterKeys.OUT_DIR, "./")
        seed = general_parameters.get(ParameterKeys.SEED, 42)
        config_name = general_parameters.get(ParameterKeys.CONFIG_NAME, "logmap")
        self.out_dir = f"{out_dir}/{config_name}_{seed}"
        os.makedirs(self.out_dir, exist_ok=True)
        seed_everything(seed)

    def _init_model(self) -> None:
        model_parameters = self.parameters.get(ParameterKeys.MODEL, dict())
        self.jar_path = model_parameters.get(ParameterKeys.ENCODER)

    def _init_encoder(self) -> None:
        pass

    def _init_datasets(self) -> None:
        # load just the source, target, and reference alignment files
        dataset_parameters = self.parameters.get(ParameterKeys.DATA, dict())
        self.source = dataset_parameters.get(ParameterKeys.INPUT_KG)
        self.target = dataset_parameters.get(ParameterKeys.SCHEMA)
        self.alignment = dataset_parameters.get(ParameterKeys.ALIGNMENT)

    def _init_evaluation(self) -> None:
        pass

    def _init_metrics(self) -> None:
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.f1_score = BinaryF1Score()

    def _parse_logmap_alignments(self, xml_path: str) -> List[Dict]:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        ns = {
            "align": "http://knowledgeweb.semanticweb.org/heterogeneity/alignment",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        }

        predictions = []
        for cell in root.findall(".//align:Cell", ns):
            e1_node = cell.find("align:entity1", ns)
            e2_node = cell.find("align:entity2", ns)

            if e1_node is None or e2_node is None:
                continue

            e1 = e1_node.attrib.get(f'{{{ns["rdf"]}}}resource')
            e2 = e2_node.attrib.get(f'{{{ns["rdf"]}}}resource')

            if not e1 or not e2:
                continue

            measure_node = cell.find("align:measure", ns)
            measure = float(measure_node.text) if measure_node is not None else 1.0

            src_id = e1.split("#")[-1]
            tgt_id = e2.split("#")[-1]

            predictions.append({"source": src_id, "target": tgt_id, "score": measure})
        return predictions
    def compute_metrics(self, predicts: List[Dict], references: List[Dict]) -> Dict:
        llm_confidence_th = 0.5
        pos_pairs = set((str(r["source"]), str(r["target"])) for r in references)

        evaluated_pairs = {}
        for p in predicts:
            score = float(p.get("score", 1.0))
            evaluated_pairs[(str(p["source"]), str(p["target"]))] = score

        for pair in pos_pairs:
            if pair not in evaluated_pairs:
                evaluated_pairs[pair] = 0.0

        node_y_true, node_y_pred = [], []
        edge_y_true, edge_y_pred = [], []

        for (src, tgt), score in evaluated_pairs.items():
            is_pos = 1 if (src, tgt) in pos_pairs else 0
            pred = 1 if score >= llm_confidence_th else 0

            if src.startswith('Q'):
                node_y_true.append(is_pos)
                node_y_pred.append(pred)
            else:
                edge_y_true.append(is_pos)
                edge_y_pred.append(pred)

        results = {}
        
        def compute_partition(y_pred, y_true, prefix):
            if len(y_true) == 0:
                return {f"{prefix}_precision": 0.0, f"{prefix}_recall": 0.0, f"{prefix}_f-score": 0.0, f"{prefix}_evaluated_pairs": 0}
            
            pred_t = torch.tensor(y_pred)
            true_t = torch.tensor(y_true)
            
            p = self.precision(pred_t, true_t).item() * 100
            r = self.recall(pred_t, true_t).item() * 100
            f1 = self.f1_score(pred_t, true_t).item() * 100
            
            self.precision.reset()
            self.recall.reset()
            self.f1_score.reset()
            
            return {
                f"{prefix}_precision": p,
                f"{prefix}_recall": r,
                f"{prefix}_f-score": f1,
                f"{prefix}_evaluated_pairs": len(y_true)
            }

        results.update(compute_partition(node_y_pred, node_y_true, "node"))
        results.update(compute_partition(edge_y_pred, edge_y_true, "edge"))
        
        global_y_pred = node_y_pred + edge_y_pred
        global_y_true = node_y_true + edge_y_true
        results.update(compute_partition(global_y_pred, global_y_true, "global"))

        return results

    def launch(self) -> Any:
        print("\nStarting LogMap Experiment Run...")
        print(f"Output directory: {self.out_dir}")

        summary_metrics = {}

        # TODO: just launch logmap on the dataset, then compute the metrics and save them
        start_time = time.time()
        print("\tRunning LogMap Java executable...")
        subprocess.run(
            [
                "java",
                "--add-opens",
                "java.base/java.lang=ALL-UNNAMED",
                "-jar",
                self.jar_path,
                "MATCHER",
                f"file://{os.path.abspath(self.source)}",
                f"file://{os.path.abspath(self.target)}",
                os.path.abspath(self.out_dir),
                "false",
            ],
            check=False,
            capture_output=True,
        )
        elapsed = time.time() - start_time

        align_file = os.path.join(self.out_dir, "logmap2_mappings.rdf")
        if not os.path.exists(align_file):
            print(f"\tError: LogMap failed to produce {align_file}")
            raise FileNotFoundError(f"LogMap output file not found: {align_file}")

        print("\tParsing LogMap predictions...")
        predicts = self._parse_logmap_alignments(align_file)
        print(f"\tPredictions generated: {len(predicts)} in {elapsed:.2f}s")

        save_json(predicts, os.path.join(self.out_dir, "logmap_predictions.json"))

        print("\tEvaluating predictions...")
        if self.alignment.endswith(".rdf") or self.alignment.endswith(".xml"):
            references_raw = self._parse_logmap_alignments(self.alignment)
        else:
            references_raw = load_json(self.alignment)
        
        summary_metrics = self.compute_metrics(predicts, references_raw)
        
        save_json(summary_metrics, os.path.join(self.out_dir, "summary_metrics.json"))
        print(
            f"\nExperiment complete. Summary metrics saved in {os.path.join(self.out_dir, 'summary_metrics.json')}"
        )

        return summary_metrics
