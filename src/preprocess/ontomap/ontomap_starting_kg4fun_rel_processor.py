from .ontomap_starting_kg4fun_class_processor import OntomapStartingKG4FUNClassProcessor
from src.utils import save_json, load_json
from .ontology_fields import OntologyFields
from .kg4fun_fields import RelFields
from src.utils import ParameterKeys


class OntomapStartingKG4FUNRelProcessor(OntomapStartingKG4FUNClassProcessor):
    def _init_general(self):
        super()._init_general()
        general_parameters = self.parameters.get(ParameterKeys.GENERAL, dict())
        self.filtered = general_parameters.get(ParameterKeys.FILTERED, True)

    def save(self, processed_ontology: list):
        output_file_path = f"{self.out_dir}/processed_rels.json"
        save_json(processed_ontology, output_file_path)

    def _init_data(self):
        dataset_parameters = self.parameters.get(ParameterKeys.DATA, dict())
        self.input_ontology = load_json(dataset_parameters.get(ParameterKeys.NAME))
        self.node_types = load_json(dataset_parameters.get(ParameterKeys.NODE_TYPES))
        self.edge_types = load_json(dataset_parameters.get(ParameterKeys.EDGE_TYPES))
        self.edge_type_info = load_json(
            dataset_parameters.get(ParameterKeys.EDGE_TYPE_INFO)
        )
        if self.filtered:
            self.edge_types = [
                edge_type for edge_type in self.edge_types if edge_type["target"] == "1"
            ]

    def process_rel_info(self, idx: int, info: dict):
        head, pid, tail = (
            info.get(RelFields.HEAD_CLS),
            info.get(RelFields.PID),
            info.get(RelFields.TAIL_CLS),
        )
        return {
            OntologyFields.IDX: f"rel_{idx}",
            OntologyFields.NAME: f"{head}_{pid}_{tail}",
            OntologyFields.IRI: "",
            OntologyFields.LABEL: f"{self.get_class_label(head)}-{self.get_rel_label(pid)}-{self.get_class_label(tail)}",
            OntologyFields.COMMENT: f"{self.get_class_description(head)}-{self.get_rel_description(pid)}-{self.get_class_description(tail)}",
            OntologyFields.SYNONYMS: list(),
            OntologyFields.CHILDRENS: list(),
            OntologyFields.PARENTS: list(),
}
    
    def get_class_qid(self, cls_idx: str):
        return list(filter(lambda x: x[1]["cls_idx"] == cls_idx, self.node_types.items()))[0][0]

    def get_class_label(self, cls_idx: str):
        qid = self.get_class_qid(cls_idx)
        class_info = [info for info in self.input_ontology if qid in info.keys()][0][qid]
        return class_info.get(OntologyFields.LABEL)
    
    def get_class_description(self, cls_idx: str):
        qid = self.get_class_qid(cls_idx)
        class_info = [info for info in self.input_ontology if qid in info.keys()][0][qid]
        return class_info.get(OntologyFields.COMMENT)
    
    def get_rel_label(self, pid: str):
        return self.edge_type_info.get(pid).get(RelFields.ITEM_LABEL, "")
    
    def get_rel_description(self, pid: str):
        return self.edge_type_info.get(pid).get(RelFields.ITEM_DESCRIPTION, "")

    def __call__(self):
        pbar = self.get_pbar(
            enumerate(self.edge_types),
            total=len(self.edge_types),
            desc="Processing Relations",
        )
        processed_ontology = list()
        for idx, info in pbar:
            rel_processed_info = self.process_rel_info(idx, info)
            processed_ontology.append({rel_processed_info.get(OntologyFields.NAME): rel_processed_info})
        self.save(processed_ontology)
