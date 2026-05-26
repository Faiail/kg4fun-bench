from .ontomap_starting_kg4fun_class_processor import OntomapStartingKG4FUNClassProcessor
from src.utils import ParameterKeys
from .kg4fun_fields import ClassFields
from .ontology_fields import OntologyFields


class OntomapSchemaKG4FUNClassProcessor(OntomapStartingKG4FUNClassProcessor):
    def __call__(self):
        pbar = self.get_pbar(
            self.dataset,
            total=len(self.dataset),
            desc="Processing Schema Classes",
        )
        processed_ontology = list()
        for info in pbar:
            class_processed_info = {
                OntologyFields.NAME: info.get(ClassFields.IDX, ""),
                OntologyFields.IRI: "",
                OntologyFields.LABEL: info.get(ClassFields.LABEL, ""),
                OntologyFields.COMMENT: info.get(ClassFields.DESCRIPTION, ""),
                OntologyFields.SYNONYMS: list(),
                OntologyFields.CHILDRENS: list(),
                OntologyFields.PARENTS: list(),
            }
            processed_ontology.append(class_processed_info)
        self.save(processed_ontology)