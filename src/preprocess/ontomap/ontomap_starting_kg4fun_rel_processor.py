from .ontomap_starting_kg4fun_class_processor import OntomapStartingKG4FUNClassProcessor
from src.utils import save_json
from .ontology_fields import OntologyFields
from .kf4fun_fields import ClassFields


class OntomapStartingKG4FUNRelProcessor(OntomapStartingKG4FUNClassProcessor):
    def save(self, processed_ontology: list):
        output_file_path = f"{self.out_dir}/processed_rels.json"
        save_json(processed_ontology, output_file_path)

def __call__(self):
        pbar = self.get_pbar(
            self.dataset.items(),
            population=len(self.dataset),
            desc="Processing relations",
        )
        processed_ontology = list()
        for qid, info in pbar:
            rel_processed_info = {
                OntologyFields.NAME: qid,
                OntologyFields.IRI: f"https://www.wikidata.org/wiki/Property:{qid}",
                OntologyFields.LABEL: info.get(ClassFields.LABEL, ""),
                OntologyFields.COMMENT: info.get(ClassFields.DESCRIPTION, ""),
                OntologyFields.SYNONYMS: list(),
                OntologyFields.CHILDRENS: list(),
                OntologyFields.PARENTS: list(),
            }
            processed_ontology.append(rel_processed_info)
        self.save(processed_ontology)