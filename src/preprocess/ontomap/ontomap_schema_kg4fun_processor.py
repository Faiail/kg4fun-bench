from .ontomap_starting_kg4fun_processor import OntomapStartingKG4FUNProcessor
from .ontomap_schema_kg4fun_class_processor import OntomapSchemaKG4FUNClassProcessor
from .ontomap_schema_kg4fun_rel_processor import OntomapSchemaKG4FUNRelProcessor
from src.utils import ParameterKeys


class OntomapSchemaKG4FUNProcessor(OntomapStartingKG4FUNProcessor):
    def _init_class_processor(self):
        class_processor_parameters = self.parameters.get(ParameterKeys.CLASS)
        self.class_processor = OntomapSchemaKG4FUNClassProcessor(class_processor_parameters)

    def _init_rel_processor(self):
        rel_processor_parameters = self.parameters.get(ParameterKeys.REL)
        self.rel_processor = OntomapSchemaKG4FUNRelProcessor(rel_processor_parameters)