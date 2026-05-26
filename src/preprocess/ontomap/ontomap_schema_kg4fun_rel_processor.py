from .ontomap_starting_kg4fun_rel_processor import OntomapStartingKG4FUNRelProcessor
from .kg4fun_fields import ClassFields
from .ontology_fields import OntologyFields


class OntomapSchemaKG4FUNRelProcessor(OntomapStartingKG4FUNRelProcessor):
    def __call__(self):
        pbar = self.get_pbar(
            self.dataset,
            total=len(self.dataset),
            desc="Processing Schema Relations",
        )
        processed_ontology = list()
        for info in pbar:
            rel_processed_info = {
                OntologyFields.NAME: info.get(ClassFields.IDX, ""),
                OntologyFields.IRI: "",
                OntologyFields.LABEL: info.get(ClassFields.LABEL, ""),
                OntologyFields.COMMENT: info.get(ClassFields.DESCRIPTION, ""),
                OntologyFields.SYNONYMS: list(),
                OntologyFields.CHILDRENS: list(),
                OntologyFields.PARENTS: list(),
            }
            processed_ontology.append(rel_processed_info)
        self.save(processed_ontology)