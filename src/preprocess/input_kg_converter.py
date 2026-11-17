import os
import glob
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from src.utils import ParameterKeys, load_json, save_json


class InputKGConverter:
    """
    Converts a KG4Fun input KG into the standard graph representation:
    - entities/Node.csv
    - entities/Edge.csv
    - entities/MetaEdge.csv
    - links/Node__head__Edge.csv
    - links/Edge__tail__Node.csv
    - links/Edge__is_a__MetaEdge.csv
    - entity_properties.json
    - annotations/node_annotations.json (pointing to schema Class.csv rows)
    - annotations/edge_annotations.json (pointing to schema Relation.csv rows)
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
        self.out_dir = general_parameters.get(ParameterKeys.OUT_DIR, "./input_kg")
        self.entities_dir = os.path.join(self.out_dir, "entities")
        self.links_dir = os.path.join(self.out_dir, "links")
        self.annotations_dir = os.path.join(self.out_dir, "annotations")
        os.makedirs(self.entities_dir, exist_ok=True)
        os.makedirs(self.links_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)
        self.pbar = general_parameters.get(ParameterKeys.PBAR, True)

    def _init_data(self) -> None:
        dataset_parameters = self.parameters.get(ParameterKeys.DATA, dict())
        self.data_dir = dataset_parameters.get(ParameterKeys.DATA_DIR, "./")
        self.schema_dir = dataset_parameters.get(ParameterKeys.SCHEMA_DIR)
        if not self.schema_dir or not os.path.exists(self.schema_dir):
            raise ValueError(f"Valid schema_dir is required, got: {self.schema_dir}")

        # 1. Load schema mappings
        schema_mapping_path = os.path.join(self.schema_dir, "schema_mapping.json")
        if not os.path.exists(schema_mapping_path):
            raise FileNotFoundError(f"schema_mapping.json not found in {self.schema_dir}")
        print(f"Loading schema mapping from {schema_mapping_path}...")
        self.schema_mapping = load_json(schema_mapping_path)
        self.cls2idx = self.schema_mapping.get("cls2idx", {})
        self.rel2idx = self.schema_mapping.get("rel2idx", {})
        self.tuple2comp = self.schema_mapping.get("tuple2comp", {})

        # 2. Load nodes
        nodes_path = os.path.join(self.data_dir, f"{ParameterKeys.NODES}.json")
        print(f"Loading nodes from {nodes_path}...")
        self.nodes_data = load_json(nodes_path) if os.path.exists(nodes_path) else []

        # 3. Load node info
        node_info_path = os.path.join(self.data_dir, f"{ParameterKeys.NODE_INFO}.json")
        print(f"Loading node_info from {node_info_path}...")
        self.node_info = load_json(node_info_path) if os.path.exists(node_info_path) else {}

        # 4. Load edge info
        edge_info_path = self._get_edge_info_path()
        print(f"Loading edge_info from {edge_info_path}...")
        self.edge_info = load_json(edge_info_path) if edge_info_path and os.path.exists(edge_info_path) else {}

        # 5. Load edges from partition files
        print("Loading edges from partition files...")
        self.raw_edges = self._load_edge_partitions()
        print(f"Loaded {len(self.raw_edges)} raw edge instances.")

    def _get_edge_info_path(self) -> str | None:
        candidates = [
            os.path.join(self.data_dir, ParameterKeys.EDGES, f"{ParameterKeys.EDGE_INFO}.json"),
            os.path.join(self.data_dir, f"{ParameterKeys.EDGE_INFO}.json"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def _load_edge_partitions(self) -> list[dict]:
        partition_dir = os.path.join(self.data_dir, ParameterKeys.EDGES, ParameterKeys.PARTITION)
        if not os.path.exists(partition_dir):
            partition_dir = os.path.join(self.data_dir, ParameterKeys.PARTITION)

        if not os.path.exists(partition_dir):
            return []

        partition_files = sorted(glob.glob(os.path.join(partition_dir, "*.json")))
        edges = []
        iter_files = tqdm(partition_files, desc="Reading edge partition files") if self.pbar else partition_files
        for pf in iter_files:
            try:
                data = load_json(pf)
                if isinstance(data, list):
                    edges.extend(data)
                elif isinstance(data, dict):
                    edges.extend(data.values())
            except Exception as e:
                print(f"Warning: could not read {pf}: {e}")
        return edges

    def process_entities_and_links(
        self,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        dict[str, dict],
        dict[str, int],
        dict[str, int],
        dict[tuple[str, str, str], int],
    ]:
        # 1. Collect all unique node QIDs
        unique_qids = set()
        for n in self.nodes_data:
            qid = n.get(ParameterKeys.QID)
            if qid:
                unique_qids.add(qid)

        for e in self.raw_edges:
            hq = e.get(ParameterKeys.HEAD_QID)
            tq = e.get(ParameterKeys.TAIL_QID)
            if hq:
                unique_qids.add(hq)
            if tq:
                unique_qids.add(tq)

        sorted_qids = sorted(unique_qids)
        node2idx = {qid: i for i, qid in enumerate(sorted_qids)}
        node_df = pd.DataFrame([{"idx": i, "qid": qid} for i, qid in enumerate(sorted_qids)])

        # 2. Collect unique PIDs
        unique_pids = set()
        for e in self.raw_edges:
            et = e.get(ParameterKeys.EDGE_TYPE, {})
            pid = et.get(ParameterKeys.PID) if isinstance(et, dict) else (et[1] if isinstance(et, list) and len(et) > 1 else None)
            if pid:
                unique_pids.add(pid)

        sorted_pids = sorted(unique_pids)
        meta_edge2idx = {pid: i for i, pid in enumerate(sorted_pids)}
        meta_edge_df = pd.DataFrame([{"idx": i, "pid": pid} for i, pid in enumerate(sorted_pids)])

        # 3. Deduplicate unique physical triples: (head_qid, pid, tail_qid)
        unique_triples = set()
        for e in self.raw_edges:
            hq = e.get(ParameterKeys.HEAD_QID)
            tq = e.get(ParameterKeys.TAIL_QID)
            et = e.get(ParameterKeys.EDGE_TYPE, {})
            pid = et.get(ParameterKeys.PID) if isinstance(et, dict) else (et[1] if isinstance(et, list) and len(et) > 1 else None)
            if hq and tq and pid:
                unique_triples.add((hq, pid, tq))

        sorted_triples = sorted(unique_triples)
        triple2edge_idx = {triple: i for i, triple in enumerate(sorted_triples)}
        edge_df = pd.DataFrame([{"idx": i, "edge": f"edge_{i}"} for i in range(len(sorted_triples))])

        # 4. Build Links
        node_head_edge = []
        edge_tail_node = []
        edge_is_a_meta_edge = []

        for triple, e_idx in triple2edge_idx.items():
            hq, pid, tq = triple
            if hq in node2idx:
                node_head_edge.append({"h": node2idx[hq], "t": e_idx})
            if tq in node2idx:
                edge_tail_node.append({"h": e_idx, "t": node2idx[tq]})
            if pid in meta_edge2idx:
                edge_is_a_meta_edge.append({"h": e_idx, "t": meta_edge2idx[pid]})

        head_df = pd.DataFrame(node_head_edge) if node_head_edge else pd.DataFrame(columns=["h", "t"])
        tail_df = pd.DataFrame(edge_tail_node) if edge_tail_node else pd.DataFrame(columns=["h", "t"])
        is_a_df = pd.DataFrame(edge_is_a_meta_edge) if edge_is_a_meta_edge else pd.DataFrame(columns=["h", "t"])

        # 5. Build entity_properties
        properties = {}
        for qid in sorted_qids:
            info = self.node_info.get(qid, {})
            properties[qid] = {
                "label": info.get("itemLabel", "Unknown Label"),
                "description": info.get("itemDescription", "-"),
            }

        for pid in sorted_pids:
            info = self.edge_info.get(pid, {})
            properties[pid] = {
                "label": info.get("itemLabel", f"Relation {pid}"),
                "description": info.get("itemDescription", "-"),
            }

        for triple, e_idx in triple2edge_idx.items():
            hq, pid, tq = triple
            info = self.edge_info.get(pid, {})
            properties[f"edge_{e_idx}"] = {
                "label": info.get("itemLabel", f"Relation {pid}"),
                "description": info.get("itemDescription", "-"),
                "pid": pid,
                "head_qid": hq,
                "tail_qid": tq,
            }

        return (
            node_df,
            edge_df,
            meta_edge_df,
            head_df,
            tail_df,
            is_a_df,
            properties,
            node2idx,
            meta_edge2idx,
            triple2edge_idx,
        )

    def process_annotations(
        self,
        node2idx: dict[str, int],
        triple2edge_idx: dict[tuple[str, str, str], int],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        prune_cls_target = int(self.cls2idx.get("-1", len(self.cls2idx)))
        prune_rel_target = int(self.rel2idx.get("-1", len(self.rel2idx)))

        # 1. Node Annotations: (input_kg_node_idx, schema_class_idx)
        node_pairs = set()
        for n in self.nodes_data:
            qid = n.get(ParameterKeys.QID)
            cls_idx = n.get(ParameterKeys.CLS_IDX)
            if qid is None or cls_idx is None or qid not in node2idx:
                continue

            node_idx = node2idx[qid]
            if cls_idx == -1 or str(cls_idx) == "-1":
                target_cls = prune_cls_target
            else:
                str_cls = str(cls_idx)
                target_cls = int(self.cls2idx.get(str_cls, prune_cls_target))

            node_pairs.add((node_idx, target_cls))

        # Ensure all nodes in node2idx have at least one annotation
        annotated_nodes = {p[0] for p in node_pairs}
        for n_idx in range(len(node2idx)):
            if n_idx not in annotated_nodes:
                node_pairs.add((n_idx, prune_cls_target))

        sorted_node_pairs = sorted(node_pairs, key=lambda x: (x[0], x[1]))
        node_annot_df = pd.DataFrame(
            sorted_node_pairs, columns=["input_kg_node_idx", "schema_class_idx"]
        )

        # 2. Edge Annotations: (input_kg_edge_idx, schema_relation_idx)
        edge_pairs = set()
        for e in self.raw_edges:
            hq = e.get(ParameterKeys.HEAD_QID)
            tq = e.get(ParameterKeys.TAIL_QID)
            et = e.get(ParameterKeys.EDGE_TYPE, {})
            if isinstance(et, dict):
                pid = et.get(ParameterKeys.PID)
                head_cls = et.get(ParameterKeys.HEAD_CLS)
                tail_cls = et.get(ParameterKeys.TAIL_CLS)
            elif isinstance(et, list) and len(et) == 3:
                head_cls, pid, tail_cls = et
            else:
                continue

            triple = (hq, pid, tq)
            if triple not in triple2edge_idx:
                continue

            edge_idx = triple2edge_idx[triple]
            if head_cls == -1 or tail_cls == -1 or str(head_cls) == "-1" or str(tail_cls) == "-1":
                target_rel = prune_rel_target
            else:
                key = f"{head_cls}_{pid}_{tail_cls}"
                comp_id = self.tuple2comp.get(key)
                if comp_id is not None:
                    str_comp = str(comp_id)
                    target_rel = int(self.rel2idx.get(str_comp, prune_rel_target))
                else:
                    target_rel = prune_rel_target

            edge_pairs.add((edge_idx, target_rel))

        # Ensure all edges in triple2edge_idx have at least one annotation
        annotated_edges = {p[0] for p in edge_pairs}
        for e_idx in range(len(triple2edge_idx)):
            if e_idx not in annotated_edges:
                edge_pairs.add((e_idx, prune_rel_target))

        sorted_edge_pairs = sorted(edge_pairs, key=lambda x: (x[0], x[1]))
        edge_annot_df = pd.DataFrame(
            sorted_edge_pairs, columns=["input_kg_edge_idx", "schema_relation_idx"]
        )

        return node_annot_df, edge_annot_df

    def __call__(self) -> None:
        print("1. Processing Input KG Entities and Links...")
        (
            node_df,
            edge_df,
            meta_edge_df,
            head_df,
            tail_df,
            is_a_df,
            properties,
            node2idx,
            meta_edge2idx,
            triple2edge_idx,
        ) = self.process_entities_and_links()

        # Save entities
        node_csv_path = os.path.join(self.entities_dir, "Node.csv")
        edge_csv_path = os.path.join(self.entities_dir, "Edge.csv")
        meta_edge_csv_path = os.path.join(self.entities_dir, "MetaEdge.csv")
        node_df.to_csv(node_csv_path, index=False)
        edge_df.to_csv(edge_csv_path, index=False)
        meta_edge_df.to_csv(meta_edge_csv_path, index=False)
        print(f"Saved {len(node_df)} nodes, {len(edge_df)} edges, {len(meta_edge_df)} meta-edges.")

        # Save links
        head_csv_path = os.path.join(self.links_dir, "Node__head__Edge.csv")
        tail_csv_path = os.path.join(self.links_dir, "Edge__tail__Node.csv")
        is_a_csv_path = os.path.join(self.links_dir, "Edge__is_a__MetaEdge.csv")
        head_df.to_csv(head_csv_path, index=False)
        tail_df.to_csv(tail_csv_path, index=False)
        is_a_df.to_csv(is_a_csv_path, index=False)
        print(f"Saved {len(head_df)} head links, {len(tail_df)} tail links, {len(is_a_df)} is_a links.")

        # Save entity_properties.json
        props_path = os.path.join(self.out_dir, "entity_properties.json")
        save_json(properties, props_path)
        print(f"Saved {len(properties)} entity properties to {props_path}")

        # 2. Process and save annotations (1-to-1 rows for easy tensor/PyG computations)
        print("2. Generating Ground-Truth Annotations using Schema Indices...")
        node_annot_df, edge_annot_df = self.process_annotations(node2idx, triple2edge_idx)

        node_annot_csv_path = os.path.join(self.annotations_dir, "node_annotations.csv")
        edge_annot_csv_path = os.path.join(self.annotations_dir, "edge_annotations.csv")
        node_annot_df.to_csv(node_annot_csv_path, index=False)
        edge_annot_df.to_csv(edge_annot_csv_path, index=False)

        # Also save JSON format with list of records for convenience
        node_annot_json_path = os.path.join(self.annotations_dir, "node_annotations.json")
        edge_annot_json_path = os.path.join(self.annotations_dir, "edge_annotations.json")
        save_json(node_annot_df.to_dict(orient="records"), node_annot_json_path)
        save_json(edge_annot_df.to_dict(orient="records"), edge_annot_json_path)

        print(f"Saved {len(node_annot_df)} node annotation rows to {node_annot_csv_path}")
        print(f"Saved {len(edge_annot_df)} edge annotation rows to {edge_annot_csv_path}")

        print("Input KG conversion completed successfully!")
