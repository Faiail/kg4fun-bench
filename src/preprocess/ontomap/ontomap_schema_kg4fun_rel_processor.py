from .ontomap_schema_kg4fun_class_processor import OntomapSchemaKG4FUNClassProcessor
from .kg4fun_fields import ClassFields
from .ontology_fields import OntologyFields
from src.utils import save_json


class OntomapSchemaKG4FUNRelProcessor(OntomapSchemaKG4FUNClassProcessor):
    def save(self, processed_ontology):
        output_file_path = f"{self.out_dir}/processed_rels.json"
        save_json(processed_ontology, output_file_path)

    def __call__(self):
        pbar = self.get_pbar(
            self.dataset,
            total=len(self.dataset),
            desc="Processing Schema Relations",
        )
        processed_ontology = list()
        for info in pbar:
            rel_processed_info = {
                f"rel_{info.get(ClassFields.IDX)}": {
                    OntologyFields.NAME: info.get(ClassFields.IDX, ""),
                    OntologyFields.IRI: "",
                    OntologyFields.LABEL: info.get(ClassFields.LABEL, ""),
                    OntologyFields.COMMENT: info.get(ClassFields.DESCRIPTION, ""),
                    OntologyFields.SYNONYMS: list(),
                    OntologyFields.CHILDRENS: list(),
                    OntologyFields.PARENTS: list(),
                }
            }
            processed_ontology.append(rel_processed_info)
        prune_rel_info = {
            "rel_-1": {
                OntologyFields.NAME: -1,
                OntologyFields.IRI: "",
                OntologyFields.LABEL: "PruneEdge",
                OntologyFields.COMMENT: "An edge which is out of domain and must be pruned.",
                OntologyFields.SYNONYMS: list(),
                OntologyFields.CHILDRENS: list(),
                OntologyFields.PARENTS: list(),
            }
        }
        processed_ontology.append(prune_rel_info)
        self.save(processed_ontology)
