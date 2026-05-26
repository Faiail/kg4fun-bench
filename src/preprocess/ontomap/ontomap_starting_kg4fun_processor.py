from .ontomap_starting_kg4fun_class_processor import OntomapStartingKG4FUNClassProcessor
from .ontomap_starting_kg4fun_rel_processor import OntomapStartingKG4FUNRelProcessor
from src.utils import load_json, save_json, ParameterKeys
import os


class OntomapStartingKG4FUNProcessor:
    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.init()

    def init(self):
        print("Init general")
        self._init_general()
        print("Init class processor")
        self._init_class_processor()

    def _init_general(self):
        general_parameters = self.parameters.get(ParameterKeys.GENERAL, dict())
        self.out_dir = general_parameters.get(ParameterKeys.OUT_DIR, "./")
        os.makedirs(self.out_dir, exist_ok=True)

    def _init_class_processor(self):
        class_processor_parameters = self.parameters.get(ParameterKeys.CLASS)
        self.class_processor = OntomapStartingKG4FUNClassProcessor(class_processor_parameters)

    def _init_rel_processor(self):
        rel_processor_parameters = self.parameters.get(ParameterKeys.REL)
        self.rel_processor = OntomapStartingKG4FUNRelProcessor(rel_processor_parameters)

    def __call__(self):
        self.class_processor()
        self._init_rel_processor()
        self.rel_processor()
        # load both ontologies
        classes_file_path = f"{self.class_processor.out_dir}/processed_classes.json"
        rels_file_path = f"{self.rel_processor.out_dir}/processed_rels.json"
        classes = load_json(classes_file_path)
        rels = load_json(rels_file_path)
        # save the whole ontology
        output_file_path = f"{self.out_dir}/processed_ontology.json"
        save_json(classes + rels, output_file_path)
        
