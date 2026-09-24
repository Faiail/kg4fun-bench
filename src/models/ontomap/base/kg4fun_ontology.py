import os
from typing import Any, Dict, List, Tuple
from src.models.ontomap.base.ontology import BaseAlignmentsParser, BaseOntologyParser
from src.models.ontomap.utils import io


class KG4FunOntology(BaseOntologyParser):
    def __init__(self, ontology_file_name: str = "processed_ontology.json"):
        super().__init__()
        self.ontology_file_name = ontology_file_name

    def load_ontology(self, input_file_path: str) -> Any:
        return io.read_json(input_file_path)

    def get_owl_classes(self, ontology: Any) -> List[Tuple[str, Dict]]:
        if isinstance(ontology, list):
            classes = []
            for item in ontology:
                if isinstance(item, dict):
                    for k, v in item.items():
                        classes.append((k, v))
                elif isinstance(item, (tuple, list)) and len(item) == 2:
                    classes.append(tuple(item))
            return classes
        elif isinstance(ontology, dict):
            return list(ontology.items())
        return ontology

    def is_contain_label(self, owl_class: Tuple[str, Dict]) -> bool:
        return True

    def get_name(self, owl_class: Tuple[str, Dict]) -> str:
        if isinstance(owl_class, (tuple, list)) and len(owl_class) == 2:
            return str(owl_class[1].get("name", owl_class[0]))
        return str(owl_class)

    def get_iri(self, owl_class: Tuple[str, Dict]) -> str:
        if isinstance(owl_class, (tuple, list)) and len(owl_class) == 2:
            return str(owl_class[0])
        return str(owl_class)

    def get_label(self, owl_class: Tuple[str, Dict]) -> str:
        if isinstance(owl_class, (tuple, list)) and len(owl_class) == 2:
            label = owl_class[1].get("label", "")
            return str(label) if label else str(owl_class[0])
        return ""

    def get_comments(self, owl_class: Tuple[str, Dict]) -> str:
        if isinstance(owl_class, (tuple, list)) and len(owl_class) == 2:
            return str(owl_class[1].get("comment", owl_class[1].get("description", "")))
        return ""

    def get_synonyms(self, owl_class: Tuple[str, Dict]) -> List:
        if isinstance(owl_class, (tuple, list)) and len(owl_class) == 2:
            return owl_class[1].get("synonyms", list())
        return list()

    def get_childrens(self, owl_class: Tuple[str, Dict]) -> List:
        if isinstance(owl_class, (tuple, list)) and len(owl_class) == 2:
            return owl_class[1].get("childrens", list())
        return list()

    def get_parents(self, owl_class: Tuple[str, Dict]) -> List:
        if isinstance(owl_class, (tuple, list)) and len(owl_class) == 2:
            return owl_class[1].get("parents", list())
        return list()

    def parse(self, root_dir: str, ontology_file_name: str = None) -> List[Dict]:
        file_name = (
            ontology_file_name
            if ontology_file_name is not None
            else self.ontology_file_name
        )
        input_file_path = os.path.join(root_dir, file_name)
        print(f"\t\tworking on {input_file_path}")
        ontology = self.load_ontology(input_file_path=input_file_path)
        return self.extract_data(ontology)


class KG4FunAlignments(BaseAlignmentsParser):
    def __init__(
        self,
        reference_file_name: str = "alignment/alignment.json",
        only_positive: bool = True,
    ):
        super().__init__()
        self.reference_file_name = reference_file_name
        self.only_positive = only_positive

    def load_ontology(self, input_file_path: str) -> Any:
        return io.read_json(input_file_path)

    def _normalize_target(self, src: str, tgt: Any) -> str:
        tgt_str = str(tgt)
        if not (tgt_str.startswith("cls_") or tgt_str.startswith("rel_")):
            if str(src).startswith("Q"):
                tgt_str = f"cls_{tgt_str}"
            else:
                tgt_str = f"rel_{tgt_str}"
        return tgt_str

    def extract_data(self, reference: Any) -> List[Dict]:
        return self._extract_alignments(reference, only_positive=self.only_positive)

    def _extract_alignments(
        self, reference: Any, only_positive: bool = False, only_negative: bool = False
    ) -> List[Dict]:
        parsed_references = []
        if isinstance(reference, list):
            for item in reference:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    src, tgt, rel = item[0], item[1], item[2]
                elif isinstance(item, dict):
                    src = item.get("source") or item.get("src")
                    tgt = item.get("target") or item.get("tgt")
                    rel = item.get("relation") or item.get("rel") or item.get("label", "yes")
                else:
                    continue

                rel_str = str(rel).lower()
                if only_positive and rel_str != "yes":
                    continue
                if only_negative and rel_str != "no":
                    continue

                tgt_normalized = self._normalize_target(src=str(src), tgt=tgt)
                parsed_references.append(
                    {
                        "source": str(src),
                        "target": tgt_normalized,
                        "relation": str(rel),
                    }
                )
        return parsed_references

    def get_positive(self, reference: Any) -> List[Dict]:
        return self._extract_alignments(reference, only_positive=True)

    def get_negative(self, reference: Any) -> List[Dict]:
        return self._extract_alignments(reference, only_negative=True)

    def get_all(self, reference: Any) -> List[Dict]:
        return self._extract_alignments(reference, only_positive=False, only_negative=False)

    def parse(self, root_dir: str, reference_file_name: str = None) -> List[Dict]:
        file_name = (
            reference_file_name
            if reference_file_name is not None
            else self.reference_file_name
        )
        input_file_path = os.path.join(root_dir, file_name)
        print(f"\t\tworking on reference: {input_file_path}")
        reference = self.load_ontology(input_file_path=input_file_path)
        return self.extract_data(reference)