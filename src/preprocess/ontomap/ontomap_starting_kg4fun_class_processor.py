from src.utils import load_json, save_json
from src.utils import ParameterKeys
from tqdm import tqdm
from .ontology_fields import OntologyFields
from .kg4fun_fields import ClassFields
import os


class OntomapStartingKG4FUNClassProcessor:
    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.init()

    def init(self):
        print("Init general parameters")
        self._init_general()
        print("Init data parameters")
        self._init_data()

    def _init_general(self):
        general_parameters = self.parameters.get(ParameterKeys.GENERAL, dict())
        self.out_dir = general_parameters.get(ParameterKeys.OUT_DIR, "./")
        os.makedirs(self.out_dir, exist_ok=True)
        self.pbar = general_parameters.get(ParameterKeys.PBAR, False)

    def _init_data(self):
        dataset_parameters = self.parameters.get(ParameterKeys.DATA, dict())
        fname = dataset_parameters.get(ParameterKeys.NAME)
        self.dataset = load_json(fname)

        split_dir = os.path.dirname(fname)
        self.node_types_path = os.path.join(split_dir, "node_types.json")
        self.node_info_path = os.path.join(split_dir, "node_info.json")
        self.nodes_path = os.path.join(split_dir, "nodes.json")

        self.additional_classes = {}
        if os.path.exists(self.node_types_path):
            node_types = load_json(self.node_types_path)
            for root_qid, data in node_types.items():
                for item in data.get("positive", []) + data.get("negative", []):
                    qid = item.get("QID")
                    if qid and qid not in self.dataset and qid not in self.additional_classes:
                        self.additional_classes[qid] = {
                            ClassFields.LABEL: item.get("label", ""),
                            ClassFields.DESCRIPTION: "",
                        }

        if os.path.exists(self.nodes_path) and os.path.exists(self.node_info_path):
            nodes = load_json(self.nodes_path)
            node_info = load_json(self.node_info_path)
            for n in nodes:
                if n.get("cls_idx") == -1:
                    qid = n.get("qid")
                    if qid and qid not in self.dataset and qid not in self.additional_classes:
                        info = node_info.get(qid, {})
                        self.additional_classes[qid] = {
                            ClassFields.LABEL: info.get("itemLabel", n.get("itemLabel", "")),
                            ClassFields.DESCRIPTION: info.get("itemDescription", ""),
                        }

    def get_pbar(self, population, **kwargs):
        return population if not self.pbar else tqdm(population, **kwargs)

    def save(self, processed_ontology: list):
        output_file_path = f"{self.out_dir}/processed_classes.json"
        save_json(processed_ontology, output_file_path)

    def __call__(self):
        combined = {**self.dataset, **self.additional_classes}
        pbar = self.get_pbar(
            enumerate(combined.items()),
            total=len(combined),
            desc="Processing classes",
        )
        processed_ontology = list()
        for idx, (qid, info) in pbar:
            class_processed_info = {
                qid: {
                    OntologyFields.IDX: f"cls_{idx}",
                    OntologyFields.NAME: qid,
                    OntologyFields.IRI: f"https://www.wikidata.org/wiki/{qid}",
                    OntologyFields.LABEL: info.get(ClassFields.LABEL, ""),
                    OntologyFields.COMMENT: info.get(ClassFields.DESCRIPTION, ""),
                    OntologyFields.SYNONYMS: list(),
                    OntologyFields.CHILDRENS: list(),
                    OntologyFields.PARENTS: list(),
                }
            }
            processed_ontology.append(class_processed_info)
        self.save(processed_ontology)
