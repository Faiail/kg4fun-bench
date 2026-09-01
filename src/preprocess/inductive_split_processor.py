import os
import json
import glob
import random
from collections import defaultdict
import networkx as nx
from tqdm import tqdm

from src.utils import load_json, save_json
from src.utils import ParameterKeys


class InductiveSplitProcessor:
    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.init()

    def init(self):
        print("Init general parameters")
        self._init_general()
        print("Init data parameters")
        self._init_data()

    def _init_general(self):
        general_parameters = self.parameters.get(ParameterKeys.GENERAL, dict())
        self.out_dir = general_parameters.get(
            ParameterKeys.OUT_DIR, "data/processed/kg4fun/inductive_splits"
        )
        os.makedirs(self.out_dir, exist_ok=True)
        self.pbar = general_parameters.get(ParameterKeys.PBAR, True)
        # 70% train, 10% val, 20% test by default
        self.train_ratio = general_parameters.get("train_ratio", 0.7)
        self.val_ratio = general_parameters.get("val_ratio", 0.1)

    def _init_data(self):
        dataset_parameters = self.parameters.get(ParameterKeys.DATA, dict())
        self.data_dir = dataset_parameters.get("data_dir", "data/raw/kg4fun/dataset1")

        # Load nodes
        print("Loading nodes...")
        self.nodes = load_json(os.path.join(self.data_dir, "nodes.json"))
        self.node_qids = {n["qid"] for n in self.nodes}

        # Load edges
        print("Loading edges from partitions...")
        edge_files = glob.glob(
            os.path.join(self.data_dir, "edges", "partition", "part_*.json")
        )
        self.all_edges = []

        iter_files = (
            tqdm(edge_files, desc="Loading edge partitions")
            if self.pbar
            else edge_files
        )
        for ef in iter_files:
            self.all_edges.extend(load_json(ef))

        print(f"Total edges loaded: {len(self.all_edges)}")

        # Load global schema
        self.global_node_types = load_json(
            os.path.join(self.data_dir, "node_types.json")
        )
        self.global_edge_types = load_json(
            os.path.join(self.data_dir, "edges", "edge_types.json")
        )

    def _grow_island(self, G, target_size, unassigned):
        island = set()
        while len(island) < target_size and unassigned:
            start_node = random.sample(list(unassigned), 1)[0]
            queue = [start_node]
            unassigned.remove(start_node)
            island.add(start_node)

            while queue and len(island) < target_size:
                curr = queue.pop(0)
                neighbors = list(G.neighbors(curr))
                random.shuffle(neighbors)
                for neighbor in neighbors:
                    if neighbor in unassigned:
                        unassigned.remove(neighbor)
                        island.add(neighbor)
                        queue.append(neighbor)
                        if len(island) >= target_size:
                            break
        return island

    def filter_schema(self, target_edge_types, target_node_types):
        return {
            "edge_types": [
                et
                for et in self.global_edge_types
                if (et["pid"], et["head_cls"], et["tail_cls"]) in target_edge_types
            ],
            "node_types": {
                qid: info
                for qid, info in self.global_node_types.items()
                if info["cls_idx"] in target_node_types
            },
        }

    def __call__(self):
        print("Building undirected graph for partitioning...")
        G = nx.Graph()

        iter_edges = (
            tqdm(self.all_edges, desc="Building graph") if self.pbar else self.all_edges
        )
        for e in iter_edges:
            G.add_edge(e["head_qid"], e["tail_qid"])

        total_nodes = len(G.nodes())
        unassigned = set(G.nodes())

        train_target = int(self.train_ratio * total_nodes)
        val_target = int(self.val_ratio * total_nodes)

        print("Assigning nodes to islands (Train/Val/Test)...")
        train_nodes = self._grow_island(G, train_target, unassigned)
        val_nodes = self._grow_island(G, val_target, unassigned)
        test_nodes = unassigned  # the rest

        # Include isolated nodes that weren't in any edge
        isolated = list(self.node_qids - set(G.nodes()))
        random.shuffle(isolated)

        idx1 = int(self.train_ratio * len(isolated))
        idx2 = int((self.train_ratio + self.val_ratio) * len(isolated))
        train_nodes.update(isolated[:idx1])
        val_nodes.update(isolated[idx1:idx2])
        test_nodes.update(isolated[idx2:])

        print(
            f"Nodes assigned - Train: {len(train_nodes)}, Val: {len(val_nodes)}, Test: {len(test_nodes)}"
        )

        print("Extracting independent islands (filtering edges)...")
        train_edges, val_edges, test_edges = [], [], []
        train_edge_types, val_edge_types, test_edge_types = set(), set(), set()

        # Initialize node types from the nodes themselves (covers isolated nodes)
        train_node_types = set(
            n["cls_idx"] for n in self.nodes if n["qid"] in train_nodes
        )
        val_node_types = set(n["cls_idx"] for n in self.nodes if n["qid"] in val_nodes)
        test_node_types = set(
            n["cls_idx"] for n in self.nodes if n["qid"] in test_nodes
        )

        iter_edges2 = (
            tqdm(self.all_edges, desc="Filtering edges")
            if self.pbar
            else self.all_edges
        )
        for e in iter_edges2:
            u, v = e["head_qid"], e["tail_qid"]
            if u in train_nodes and v in train_nodes:
                train_edges.append(e)
                train_edge_types.add(
                    (
                        e["edge_type"]["pid"],
                        e["edge_type"]["head_cls"],
                        e["edge_type"]["tail_cls"],
                    )
                )
                train_node_types.add(e["edge_type"]["head_cls"])
                train_node_types.add(e["edge_type"]["tail_cls"])
            elif u in val_nodes and v in val_nodes:
                val_edges.append(e)
                val_edge_types.add(
                    (
                        e["edge_type"]["pid"],
                        e["edge_type"]["head_cls"],
                        e["edge_type"]["tail_cls"],
                    )
                )
                val_node_types.add(e["edge_type"]["head_cls"])
                val_node_types.add(e["edge_type"]["tail_cls"])
            elif u in test_nodes and v in test_nodes:
                test_edges.append(e)
                test_edge_types.add(
                    (
                        e["edge_type"]["pid"],
                        e["edge_type"]["head_cls"],
                        e["edge_type"]["tail_cls"],
                    )
                )
                test_node_types.add(e["edge_type"]["head_cls"])
                test_node_types.add(e["edge_type"]["tail_cls"])

        import shutil

        # Helper to export a full raw dataset directory
        def export_raw_split(
            split_name, nodes_subset, edges_subset, edge_types_subset, node_types_subset
        ):
            split_dir = os.path.join(self.out_dir, split_name)
            edges_dir = os.path.join(split_dir, "edges")
            partition_dir = os.path.join(edges_dir, "partition")
            os.makedirs(partition_dir, exist_ok=True)

            schema = self.filter_schema(edge_types_subset, node_types_subset)

            # Filter node_type_info.json
            src_nti = os.path.join(self.data_dir, "node_type_info.json")
            if os.path.exists(src_nti):
                nti = load_json(src_nti)
                filtered_nti = {
                    k: v for k, v in nti.items() if k in schema["node_types"]
                }
                save_json(filtered_nti, os.path.join(split_dir, "node_type_info.json"))

            src_ni = os.path.join(self.data_dir, "node_info.json")
            if os.path.exists(src_ni):
                shutil.copy2(src_ni, os.path.join(split_dir, "node_info.json"))

            # Filter edge_info.json
            src_edge_info = os.path.join(self.data_dir, "edges", "edge_info.json")
            if os.path.exists(src_edge_info):
                ei = load_json(src_edge_info)
                valid_pids = {et["pid"] for et in schema["edge_types"]}
                filtered_ei = {k: v for k, v in ei.items() if k in valid_pids}
                save_json(filtered_ei, os.path.join(edges_dir, "edge_info.json"))

            # Filter node_types/summarization/.../node_type_info.json
            src_nt_dir = os.path.join(self.data_dir, "node_types")
            if os.path.isdir(src_nt_dir):
                shutil.copytree(
                    src_nt_dir,
                    os.path.join(split_dir, "node_types"),
                    dirs_exist_ok=True,
                )
                for root, _, files in os.walk(os.path.join(split_dir, "node_types")):
                    for file in files:
                        if file.endswith(".json"):
                            path = os.path.join(root, file)
                            try:
                                data = load_json(path)
                                if (
                                    isinstance(data, list)
                                    and len(data) > 0
                                    and "idx" in data[0]
                                ):
                                    filtered_data = [
                                        d for d in data if d["idx"] in node_types_subset
                                    ]
                                    save_json(filtered_data, path)
                            except:
                                pass

            # Copy edge_types/summarization/... blindly (since alignment filters via component_id)
            src_et_dir = os.path.join(self.data_dir, "edge_types")
            if os.path.isdir(src_et_dir):
                shutil.copytree(
                    src_et_dir,
                    os.path.join(split_dir, "edge_types"),
                    dirs_exist_ok=True,
                )

            # Save filtered schemas
            save_json(schema["node_types"], os.path.join(split_dir, "node_types.json"))
            save_json(schema["edge_types"], os.path.join(edges_dir, "edge_types.json"))

            # Save nodes and edges
            save_json(
                [n for n in self.nodes if n["qid"] in nodes_subset],
                os.path.join(split_dir, "nodes.json"),
            )
            save_json(edges_subset, os.path.join(partition_dir, "part_0.json"))

            # Filter connected_components and mapping based on the schemas
            src_cc = os.path.join(self.data_dir, "edges", "connected_components.json")
            if os.path.exists(src_cc):
                cc = load_json(src_cc)
                filtered_cc = []
                for comp in cc:
                    valid_ets = [
                        et
                        for et in comp.get("edge_types", [])
                        if tuple(et) in edge_types_subset
                    ]
                    if valid_ets:
                        comp_copy = comp.copy()
                        comp_copy["edge_types"] = valid_ets
                        filtered_cc.append(comp_copy)
                save_json(
                    filtered_cc, os.path.join(edges_dir, "connected_components.json")
                )

            src_mapping = os.path.join(
                self.data_dir, "edges", "edge_component_mapping.json"
            )
            if os.path.exists(src_mapping):
                mapping = load_json(src_mapping)
                filtered_mapping = [
                    m for m in mapping if tuple(m["edge_type"]) in edge_types_subset
                ]
                save_json(
                    filtered_mapping,
                    os.path.join(edges_dir, "edge_component_mapping.json"),
                )

        # For Setting 1, val/test get the global schema
        global_et = set(
            (et["pid"], et["head_cls"], et["tail_cls"]) for et in self.global_edge_types
        )
        global_nt = set(info["cls_idx"] for qid, info in self.global_node_types.items())

        # Extract the base dataset name (e.g. 'dataset1', 'dataset2') from the input directory path
        dataset_name = os.path.basename(os.path.normpath(self.data_dir))

        # Export Setting 1
        export_raw_split(
            f"{dataset_name}_setting1_train",
            train_nodes,
            train_edges,
            train_edge_types,
            train_node_types,
        )
        export_raw_split(
            f"{dataset_name}_setting1_val", val_nodes, val_edges, global_et, global_nt
        )
        export_raw_split(
            f"{dataset_name}_setting1_test",
            test_nodes,
            test_edges,
            global_et,
            global_nt,
        )

        # Export Setting 2
        export_raw_split(
            f"{dataset_name}_setting2_train",
            train_nodes,
            train_edges,
            train_edge_types,
            train_node_types,
        )
        export_raw_split(
            f"{dataset_name}_setting2_val",
            val_nodes,
            val_edges,
            val_edge_types,
            val_node_types,
        )
        export_raw_split(
            f"{dataset_name}_setting2_test",
            test_nodes,
            test_edges,
            test_edge_types,
            test_node_types,
        )

        print("Done! Full raw directory splits generated.")
