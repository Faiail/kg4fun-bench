# -*- coding: utf-8 -*-
import os
from typing import Any, Dict

from src.models.ontomap.base.dataset import OMDataset
from src.models.ontomap.base.kg4fun_ontology import KG4FunAlignments, KG4FunOntology
from src.models.ontomap.utils import io


class KG4FunOMDataset(OMDataset):
    track: str = "kg4fun"
    ontology_name: str = ""

    source_ontology: Any = KG4FunOntology(
        ontology_file_name="input_kg/processed_ontology.json"
    )
    target_ontology: Any = KG4FunOntology(
        ontology_file_name="schema/processed_ontology.json"
    )
    alignments: Any = KG4FunAlignments(
        reference_file_name="alignment/alignment.json"
    )

    working_dir: str = ""

    def collect(self, root_dir: str) -> Dict:
        om_root_path = os.path.join(root_dir, self.track, self.ontology_name)
        data = {
            "dataset-info": {"track": self.track, "ontology-name": self.ontology_name},
            "source": self._load_source_ontology(om_root_path),
            "target": self._load_target_ontology(om_root_path),
            "reference": self._load_alignments(om_root_path),
        }
        return data

    def _load_source_ontology(self, om_root_path: str):
        return self.source_ontology.parse(
            root_dir=om_root_path,
            ontology_file_name="input_kg/processed_ontology.json",
        )

    def _load_target_ontology(self, om_root_path: str):
        return self.target_ontology.parse(
            root_dir=om_root_path,
            ontology_file_name="schema/processed_ontology.json",
        )

    def _load_alignments(self, om_root_path: str):
        return self.alignments.parse(
            root_dir=om_root_path,
            reference_file_name="alignment/alignment.json",
        )

    def load_from_json(self, root_dir: str) -> Dict:
        json_file_path = os.path.join(
            root_dir, self.track, self.ontology_name, "om.json"
        )
        if os.path.exists(json_file_path):
            return io.read_json(input_path=json_file_path)
        data = self.collect(root_dir=root_dir)
        io.write_json(output_path=json_file_path, json_data=data)
        return data

    def __dir__(self):
        return os.path.join(self.track, self.ontology_name)

    def __str__(self):
        return f"{self.track}/{self.ontology_name}"