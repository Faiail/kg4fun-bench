import os
import pandas as pd
from tqdm import tqdm
from src.utils import ParameterKeys, load_json, save_json


class SchemaConverter:
    """
    Converts a KG4Fun target schema into the standard graph representation:
    - entities/Class.csv
    - entities/Relation.csv
    - links/Class__head__Relation.csv
    - links/Relation__tail__Class.csv
    - entity_properties.json
    - schema_mapping.json
    """

    def __init__(self, parameters: dict) -> None:
        self.parameters = parameters
        self.init()

    def init(self) -> None:
        print("Init general parameters")
        self._init_general()
        print("Init data parameters")
        self._init_data()

    def _init_general(self) -> None:
        general_parameters = self.parameters.get(ParameterKeys.GENERAL, dict())
        self.out_dir = general_parameters.get(ParameterKeys.OUT_DIR, "./schema")
        self.entities_dir = os.path.join(self.out_dir, "entities")
        self.links_dir = os.path.join(self.out_dir, "links")
        os.makedirs(self.entities_dir, exist_ok=True)
        os.makedirs(self.links_dir, exist_ok=True)
        self.pbar = general_parameters.get(ParameterKeys.PBAR, True)

    def _init_data(self) -> None:
        dataset_parameters = self.parameters.get(ParameterKeys.DATA, dict())
        self.data_dir = dataset_parameters.get(ParameterKeys.DATA_DIR, "./")

        # 1. Load node_type_info (classes)
        node_type_info_path = self._get_node_type_info_path()
        print(f"Loading schema classes from {node_type_info_path}...")
        self.node_type_info = load_json(node_type_info_path)

        # 2. Load edge_type_info (relations)
        edge_type_info_path = self._get_edge_type_info_path()
        print(f"Loading schema relations from {edge_type_info_path}...")
        self.edge_type_info = load_json(edge_type_info_path)

        # 3. Load connected components (schema relation topology)
        connected_components_path = self._get_connected_components_path()
        print(f"Loading schema relation components from {connected_components_path}...")
        self.connected_components = load_json(connected_components_path)

        # 4. Optional edge_component_mapping
        ecm_path = self._get_edge_component_mapping_path()
        self.edge_component_mapping = load_json(ecm_path) if ecm_path and os.path.exists(ecm_path) else None

    def _get_node_type_info_path(self) -> str:
        candidates = [
            os.path.join(self.data_dir, ParameterKeys.NODE_TYPES, "summarization", "qwen3-8b", f"{ParameterKeys.NODE_TYPE_INFO}.json"),
            os.path.join(self.data_dir, f"{ParameterKeys.NODE_TYPE_INFO}.json"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"node_type_info.json not found in {self.data_dir}")

    def _get_edge_type_info_path(self) -> str:
        candidates = [
            os.path.join(self.data_dir, ParameterKeys.EDGE_TYPES, "summarization", "qwen-8b", f"{ParameterKeys.EDGE_TYPE_INFO}.json"),
            os.path.join(self.data_dir, ParameterKeys.EDGES, f"{ParameterKeys.EDGE_TYPE_INFO}.json"),
            os.path.join(self.data_dir, f"{ParameterKeys.EDGE_TYPE_INFO}.json"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"edge_type_info.json not found in {self.data_dir}")

    def _get_connected_components_path(self) -> str:
        candidates = [
            os.path.join(self.data_dir, ParameterKeys.EDGES, f"{ParameterKeys.CONNECTED_COMPONENTS}.json"),
            os.path.join(self.data_dir, f"{ParameterKeys.CONNECTED_COMPONENTS}.json"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"connected_components.json not found in {self.data_dir}")

    def _get_edge_component_mapping_path(self) -> str | None:
        candidates = [
            os.path.join(self.data_dir, ParameterKeys.EDGES, f"{ParameterKeys.EDGE_COMPONENT_MAPPING}.json"),
            os.path.join(self.data_dir, f"{ParameterKeys.EDGE_COMPONENT_MAPPING}.json"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def _normalize_items(self, raw_data) -> list[dict]:
        if isinstance(raw_data, list):
            return raw_data
        if isinstance(raw_data, dict):
            items = []
            for k, v in raw_data.items():
                if isinstance(v, dict):
                    idx_val = v.get("idx", k)
                    items.append({"idx": int(idx_val) if str(idx_val).isdigit() else idx_val, **v})
                else:
                    items.append({"idx": int(k) if str(k).isdigit() else k, "value": v})
            return items
        raise ValueError(f"Unsupported data format: {type(raw_data)}")

    def process_classes(self) -> tuple[pd.DataFrame, dict[int, int], dict[str, dict]]:
        items = self._normalize_items(self.node_type_info)
        items_sorted = sorted(items, key=lambda x: int(x["idx"]) if str(x["idx"]).isdigit() else str(x["idx"]))

        class_rows = []
        cls2idx = {}
        class_properties = {}

        for i, item in enumerate(items_sorted):
            cls_idx = item["idx"]
            cls_id = f"class_{cls_idx}"
            cls2idx[cls_idx] = i
            cls2idx[str(cls_idx)] = i
            class_rows.append({"idx": i, "class": cls_idx})
            class_properties[cls_id] = {
                "label": item.get("itemLabel", f"Class {cls_idx}"),
                "description": item.get("itemDescription", "-"),
                "cls_idx": cls_idx,
            }

        # Explicit Prune Node class
        prune_row_idx = len(items_sorted)
        cls2idx[-1] = prune_row_idx
        cls2idx["-1"] = prune_row_idx
        class_rows.append({"idx": prune_row_idx, "class": -1})
        class_properties["class_-1"] = {
            "label": "Prune Node",
            "description": "A node which is out of domain and must be pruned.",
            "cls_idx": -1,
        }

        class_df = pd.DataFrame(class_rows)
        return class_df, cls2idx, class_properties

    def process_relations(self) -> tuple[pd.DataFrame, dict[int, int], dict[str, dict]]:
        items = self._normalize_items(self.edge_type_info)
        items_sorted = sorted(items, key=lambda x: int(x["idx"]) if str(x["idx"]).isdigit() else str(x["idx"]))

        rel_rows = []
        rel2idx = {}
        rel_properties = {}

        for i, item in enumerate(items_sorted):
            comp_id = item["idx"]
            rel_id = f"rel_{comp_id}"
            rel2idx[comp_id] = i
            rel2idx[str(comp_id)] = i
            rel_rows.append({"idx": i, "relation": comp_id})
            rel_properties[rel_id] = {
                "label": item.get("itemLabel", f"Relation {comp_id}"),
                "description": item.get("itemDescription", "-"),
                "component_id": comp_id,
            }

        # Explicit Prune Edge relation
        prune_row_idx = len(items_sorted)
        rel2idx[-1] = prune_row_idx
        rel2idx["-1"] = prune_row_idx
        rel_rows.append({"idx": prune_row_idx, "relation": -1})
        rel_properties["rel_-1"] = {
            "label": "Prune Edge",
            "description": "An edge which is out of domain and must be pruned.",
            "component_id": -1,
        }

        rel_df = pd.DataFrame(rel_rows)
        return rel_df, rel2idx, rel_properties

    def process_links(
        self,
        cls2idx: dict[int, int],
        rel2idx: dict[int, int],
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], dict[int, list[int]]]:
        head_links = []
        tail_links = []
        tuple2comp = {}
        comp2tuple = {}

        components = self._normalize_items(self.connected_components)
        pbar = tqdm(components, desc="Processing schema links") if self.pbar else components

        for comp in pbar:
            comp_id = comp.get(ParameterKeys.COMPONENT_ID)
            if comp_id is None:
                continue

            edge_types = comp.get(ParameterKeys.EDGE_TYPES, [])
            if not edge_types:
                continue

            # First edge type defines the head_cls and tail_cls for this component
            sample_et = edge_types[0]
            head_cls = sample_et[0]
            tail_cls = sample_et[2]

            comp2tuple[comp_id] = [head_cls, tail_cls]

            # Register all (head, pid, tail) mappings to this component
            for et in edge_types:
                key = f"{et[0]}_{et[1]}_{et[2]}"
                tuple2comp[key] = comp_id

            # Add links if endpoints exist in the converted Class and Relation entities
            if comp_id in rel2idx:
                r_idx = rel2idx[comp_id]
                if head_cls in cls2idx:
                    head_links.append({"h": cls2idx[head_cls], "t": r_idx})
                if tail_cls in cls2idx:
                    tail_links.append({"h": r_idx, "t": cls2idx[tail_cls]})

        # If edge_component_mapping exists, ensure complete triple-to-comp mapping
        if self.edge_component_mapping:
            ecm_items = self._normalize_items(self.edge_component_mapping)
            for item in ecm_items:
                et = item.get(ParameterKeys.EDGE_TYPE)
                comp_id = item.get(ParameterKeys.COMPONENT_ID)
                if et and comp_id is not None:
                    key = f"{et[0]}_{et[1]}_{et[2]}"
                    if key not in tuple2comp:
                        tuple2comp[key] = comp_id

        # Add prune relation link: Prune Node -> Prune Rel -> Prune Node
        if -1 in cls2idx and -1 in rel2idx:
            head_links.append({"h": cls2idx[-1], "t": rel2idx[-1]})
            tail_links.append({"h": rel2idx[-1], "t": cls2idx[-1]})

        head_df = pd.DataFrame(head_links) if head_links else pd.DataFrame(columns=["h", "t"])
        tail_df = pd.DataFrame(tail_links) if tail_links else pd.DataFrame(columns=["h", "t"])

        return head_df, tail_df, tuple2comp, comp2tuple

    def __call__(self) -> None:
        print("1. Processing Schema Classes...")
        class_df, cls2idx, class_properties = self.process_classes()
        class_csv_path = os.path.join(self.entities_dir, "Class.csv")
        class_df.to_csv(class_csv_path, index=False)
        print(f"Saved {len(class_df)} classes to {class_csv_path}")

        print("2. Processing Schema Relations...")
        rel_df, rel2idx, rel_properties = self.process_relations()
        rel_csv_path = os.path.join(self.entities_dir, "Relation.csv")
        rel_df.to_csv(rel_csv_path, index=False)
        print(f"Saved {len(rel_df)} relations to {rel_csv_path}")

        print("3. Processing Schema Links (head and tail)...")
        head_df, tail_df, tuple2comp, comp2tuple = self.process_links(cls2idx, rel2idx)
        head_csv_path = os.path.join(self.links_dir, "Class__head__Relation.csv")
        tail_csv_path = os.path.join(self.links_dir, "Relation__tail__Class.csv")
        head_df.to_csv(head_csv_path, index=False)
        tail_df.to_csv(tail_csv_path, index=False)
        print(f"Saved {len(head_df)} head links to {head_csv_path}")
        print(f"Saved {len(tail_df)} tail links to {tail_csv_path}")

        print("4. Saving entity properties...")
        all_properties = {**class_properties, **rel_properties}
        properties_path = os.path.join(self.out_dir, "entity_properties.json")
        save_json(all_properties, properties_path)
        print(f"Saved {len(all_properties)} entity properties to {properties_path}")

        print("5. Saving schema mappings for Input KG Converter...")
        mapping_data = {
            "cls2idx": {str(k): v for k, v in cls2idx.items()},
            "idx2cls": {str(v): k for k, v in cls2idx.items()},
            "rel2idx": {str(k): v for k, v in rel2idx.items()},
            "idx2rel": {str(v): k for k, v in rel2idx.items()},
            "tuple2comp": tuple2comp,
            "comp2tuple": {str(k): v for k, v in comp2tuple.items()},
        }
        mapping_path = os.path.join(self.out_dir, "schema_mapping.json")
        save_json(mapping_data, mapping_path)
        print(f"Saved schema mapping metadata to {mapping_path}")

        print("Schema conversion completed successfully!")
