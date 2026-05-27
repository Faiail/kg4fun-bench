from src.utils import load_json, save_json, ParameterKeys
from tqdm import tqdm
from itertools import product
import os
import random


class OntomapKG4FUNAlignmentProcessor:
    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.init()

    def init(self):
        print("Init general")
        self._init_general()
        print("Init ontologies")
        self._init_ontologies()
        print("Init alignment data")
        self._init_alignment()

    def _init_general(self):
        general_parameters = self.parameters.get(ParameterKeys.GENERAL, dict())
        self.out_dir = general_parameters.get(ParameterKeys.OUT_DIR, "./")
        os.makedirs(self.out_dir, exist_ok=True)
        self.pbar = general_parameters.get(ParameterKeys.PBAR, False)

    def _init_ontologies(self):
        ontology_parameters = self.parameters.get(ParameterKeys.ONTOLOGY, dict())
        self.input_kg_ontology = load_json(
            ontology_parameters.get(ParameterKeys.INPUT_KG)
        )
        self.schema_ontology = load_json(ontology_parameters.get(ParameterKeys.SCHEMA))

    def _init_alignment(self):
        alignment_parameters = self.parameters.get(ParameterKeys.ALIGNMENT, dict())
        self.node_types = load_json(alignment_parameters.get(ParameterKeys.NODE_TYPES))
        self.connected_components = load_json(
            alignment_parameters.get(ParameterKeys.CONNECTED_COMPONENTS)
        )
        self.num_class_negative = alignment_parameters.get(
            ParameterKeys.NUM_CLASS_NEGATIVE, 0
        )
        self.num_rel_negative = alignment_parameters.get(
            ParameterKeys.NUM_REL_NEGATIVE, 0
        )
        # node types (to have the information about cls idx and root + positive + negative)
        # connected component mapping (to get the information about the gt between the edge types and the schema edge types)

    def get_pbar(self, population, **kwargs):
        return population if not self.pbar else tqdm(population, **kwargs)

    def get_schema_classes(self):
        return list(
            map(
                lambda y: y[0].split("cls_")[1],
                filter(
                    lambda x: "cls_" in x[0],
                    map(lambda x: list(x.items())[0], self.schema_ontology),
                ),
            )
        )

    def get_associated_qids(self, class_idx: int):
        target_dict = {
            k: v
            for k, v in self.node_types.items()
            if v.get("cls_idx") == int(class_idx)
        }
        root_qid = list(target_dict.keys())[0]
        return (
            [root_qid]
            + list(x["QID"] for x in target_dict[root_qid].get("positive", list()))
            + list(x["QID"] for x in target_dict[root_qid].get("negative", list()))
        )

    def get_class_alignment(self):
        positive_alignments = self.get_positive_class_alignment()
        negative_alignments = self.get_negative_class_alignment(positive_alignments)
        return positive_alignments + negative_alignments

    def get_rel_alignment(self):
        positive_alignments = self.get_positive_rel_alignment()
        negative_alignments = self.get_negative_rel_alignment(positive_alignments)
        return positive_alignments + negative_alignments

    def get_positive_class_alignment(self):
        class_alignments = []
        for class_idx in self.get_pbar(
            self.get_schema_classes(), desc="Getting positive class alignment"
        ):
            class_alignments.extend(
                list(
                    map(
                        lambda qid: (qid, class_idx, "yes"),
                        self.get_associated_qids(class_idx),
                    )
                )
            )
        return class_alignments

    def get_comb_classes(self):
        start_node_types = set()
        for k, v in self.node_types.items():
            start_node_types.add(k)
            start_node_types.update(
                set(x["QID"] for x in v.get("positive", list()))
                | set(x["QID"] for x in v.get("negative", list()))
            )
        schema_node_types = set(
            map(
                lambda x: x.split("cls_")[1],
                filter(
                    lambda x: "cls_" in x,
                    map(lambda y: list(y.keys())[0], self.schema_ontology),
                ),
            )
        )
        return set(product(start_node_types, schema_node_types))

    def get_negative_class_alignment(self, positive_alignments):
        # get all possibilities (set)
        # get all positive alignments (set)
        # get the differennce and sample
        set_positive_alignments = set(map(lambda x: (x[0], x[1]), positive_alignments))
        set_comb_classes = self.get_comb_classes()
        negative_alignments = list(set_comb_classes - set_positive_alignments)
        negative_alignments = random.choices(
            negative_alignments,
            k=(
                self.num_class_negative
                if self.num_class_negative > 0
                else len(positive_alignments)
            ),
        )
        processed_negative_alignments = list()
        for negative_alignment in self.get_pbar(
            negative_alignments, desc="Getting negative class alignment"
        ):
            processed_negative_alignments.append(
                (negative_alignment[0], negative_alignment[1], "no")
            )
        return processed_negative_alignments

    def get_positive_rel_alignment(self):
        alignments = list()
        for connected_component in self.get_pbar(
            self.connected_components,
            total=len(self.connected_components),
            desc="Getting positive relation alignement",
        ):
            schema_type_idx = connected_component.get("component_id")
            for h, r, o in connected_component.get("edge_types"):
                alignments.append((f"{h}_{r}_{o}", schema_type_idx, "yes"))
        return alignments

    def get_comb_rel_parts(self):
        start_edge_types = set()
        schema_rel_ids = set()

        for connected_component in self.connected_components:
            schema_rel_ids.add(connected_component["component_id"])

            for h, r, o in connected_component.get("edge_types", []):
                start_edge_types.add(f"{h}_{r}_{o}")

        return list(start_edge_types), list(schema_rel_ids)

    def get_negative_rel_alignment(self, positive_alignments):
        positive_set = {
            (src, rel)
            for src, rel, *_ in positive_alignments
        }

        start_edge_types, schema_rel_ids = self.get_comb_rel_parts()

        target_size = (
            self.num_rel_negative
            if self.num_rel_negative > 0
            else len(positive_alignments)
        )

        negatives = set()

        # rejection sampling
        while len(negatives) < target_size:
            pair = (
                random.choice(start_edge_types),
                random.choice(schema_rel_ids),
            )

            if pair not in positive_set:
                negatives.add(pair)

        return [
            (src, rel, "no")
            for src, rel in self.get_pbar(
                negatives,
                desc="Getting negative relations alignment"
            )
        ]


    def __call__(self):
        # divide the thing in gathering the gt between classes and relations
        # for each of the two, get both positive and negative examples, equally distributed
        processed_alignment = self.get_class_alignment() + self.get_rel_alignment()
        save_json(processed_alignment, os.path.join(self.out_dir, "alignment.json"))
