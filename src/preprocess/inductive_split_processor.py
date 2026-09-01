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
        self.out_dir = general_parameters.get(ParameterKeys.OUT_DIR, "data/processed/kg4fun/inductive_splits")
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
        edge_files = glob.glob(os.path.join(self.data_dir, "edges", "partition", "part_*.json"))
        self.all_edges = []
        
        iter_files = tqdm(edge_files, desc="Loading edge partitions") if self.pbar else edge_files
        for ef in iter_files:
            self.all_edges.extend(load_json(ef))
            
        print(f"Total edges loaded: {len(self.all_edges)}")

        # Load global schema
        self.global_node_types = load_json(os.path.join(self.data_dir, "node_types.json"))
        self.global_edge_types = load_json(os.path.join(self.data_dir, "edges", "edge_types.json"))

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
            "edge_types": [et for et in self.global_edge_types if (et["pid"], et["head_cls"], et["tail_cls"]) in target_edge_types],
            "node_types": {qid: info for qid, info in self.global_node_types.items() if info["cls_idx"] in target_node_types}
        }

    def __call__(self):
        print("Building undirected graph for partitioning...")
        G = nx.Graph()
        
        iter_edges = tqdm(self.all_edges, desc="Building graph") if self.pbar else self.all_edges
        for e in iter_edges:
            G.add_edge(e["head_qid"], e["tail_qid"])
            
        total_nodes = len(G.nodes())
        unassigned = set(G.nodes())
        
        train_target = int(self.train_ratio * total_nodes)
        val_target = int(self.val_ratio * total_nodes)
        
        print("Assigning nodes to islands (Train/Val/Test)...")
        train_nodes = self._grow_island(G, train_target, unassigned)
        val_nodes = self._grow_island(G, val_target, unassigned)
        test_nodes = unassigned # the rest
        
        # Include isolated nodes that weren't in any edge
        isolated = list(self.node_qids - set(G.nodes()))
        random.shuffle(isolated)
        
        idx1 = int(self.train_ratio * len(isolated))
        idx2 = int((self.train_ratio + self.val_ratio) * len(isolated))
        train_nodes.update(isolated[:idx1])
        val_nodes.update(isolated[idx1:idx2])
        test_nodes.update(isolated[idx2:])
        
        print(f"Nodes assigned - Train: {len(train_nodes)}, Val: {len(val_nodes)}, Test: {len(test_nodes)}")
        
        print("Extracting independent islands (filtering edges)...")
        train_edges, val_edges, test_edges = [], [], []
        train_edge_types, val_edge_types, test_edge_types = set(), set(), set()
        train_node_types, val_node_types, test_node_types = set(), set(), set()
        
        iter_edges2 = tqdm(self.all_edges, desc="Filtering edges") if self.pbar else self.all_edges
        for e in iter_edges2:
            u, v = e["head_qid"], e["tail_qid"]
            if u in train_nodes and v in train_nodes:
                train_edges.append(e)
                train_edge_types.add((e["edge_type"]["pid"], e["edge_type"]["head_cls"], e["edge_type"]["tail_cls"]))
                train_node_types.add(e["edge_type"]["head_cls"])
                train_node_types.add(e["edge_type"]["tail_cls"])
            elif u in val_nodes and v in val_nodes:
                val_edges.append(e)
                val_edge_types.add((e["edge_type"]["pid"], e["edge_type"]["head_cls"], e["edge_type"]["tail_cls"]))
                val_node_types.add(e["edge_type"]["head_cls"])
                val_node_types.add(e["edge_type"]["tail_cls"])
            elif u in test_nodes and v in test_nodes:
                test_edges.append(e)
                test_edge_types.add((e["edge_type"]["pid"], e["edge_type"]["head_cls"], e["edge_type"]["tail_cls"]))
                test_node_types.add(e["edge_type"]["head_cls"])
                test_node_types.add(e["edge_type"]["tail_cls"])
                
        import shutil
        
        print("Creating schemas and full raw directories for Setting 1 and Setting 2...")
        
        # Helper to export a full raw dataset directory
        def export_raw_split(split_name, nodes_subset, edges_subset, edge_types_subset, node_types_subset):
            split_dir = os.path.join(self.out_dir, split_name)
            edges_dir = os.path.join(split_dir, "edges")
            partition_dir = os.path.join(edges_dir, "partition")
            os.makedirs(partition_dir, exist_ok=True)
            
            # Filter schema
            schema = self.filter_schema(edge_types_subset, node_types_subset)
            
            # Copy untouched info files and folders
            for fname in ["node_type_info.json", "node_info.json"]:
                src = os.path.join(self.data_dir, fname)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(split_dir, fname))
            
            for dname in ["node_types", "edge_types"]:
                src = os.path.join(self.data_dir, dname)
                if os.path.isdir(src):
                    shutil.copytree(src, os.path.join(split_dir, dname), dirs_exist_ok=True)
                    
            src_edge_info = os.path.join(self.data_dir, "edges", "edge_info.json")
            if os.path.exists(src_edge_info):
                shutil.copy2(src_edge_info, os.path.join(edges_dir, "edge_info.json"))
                
            # Save filtered schemas
            save_json(schema["node_types"], os.path.join(split_dir, "node_types.json"))
            save_json(schema["edge_types"], os.path.join(edges_dir, "edge_types.json"))
            
            # Save nodes and edges
            save_json([n for n in self.nodes if n["qid"] in nodes_subset], os.path.join(split_dir, "nodes.json"))
            save_json(edges_subset, os.path.join(partition_dir, "part_0.json"))
            
            # Filter connected_components and mapping based on the schemas
            # (Only keeping components that have at least one edge type in the filtered schema)
            src_cc = os.path.join(self.data_dir, "edges", "connected_components.json")
            if os.path.exists(src_cc):
                cc = load_json(src_cc)
                filtered_cc = []
                for comp in cc:
                    valid_ets = [et for et in comp.get("edge_types", []) if tuple(et) in edge_types_subset]
                    if valid_ets:
                        comp_copy = comp.copy()
                        comp_copy["edge_types"] = valid_ets
                        filtered_cc.append(comp_copy)
                save_json(filtered_cc, os.path.join(edges_dir, "connected_components.json"))
                
            src_mapping = os.path.join(self.data_dir, "edges", "edge_component_mapping.json")
            if os.path.exists(src_mapping):
                mapping = load_json(src_mapping)
                filtered_mapping = [m for m in mapping if tuple(m["edge_type"]) in edge_types_subset]
                save_json(filtered_mapping, os.path.join(edges_dir, "edge_component_mapping.json"))

        # For Setting 1, val/test get the global schema
        global_et = set((et["pid"], et["head_cls"], et["tail_cls"]) for et in self.global_edge_types)
        global_nt = set(info["cls_idx"] for qid, info in self.global_node_types.items())
        
        # Export Setting 1
        export_raw_split("dataset1_setting1_train", train_nodes, train_edges, train_edge_types, train_node_types)
        export_raw_split("dataset1_setting1_val", val_nodes, val_edges, global_et, global_nt)
        export_raw_split("dataset1_setting1_test", test_nodes, test_edges, global_et, global_nt)
        
        # Export Setting 2
        export_raw_split("dataset1_setting2_train", train_nodes, train_edges, train_edge_types, train_node_types)
        export_raw_split("dataset1_setting2_val", val_nodes, val_edges, val_edge_types, val_node_types)
        export_raw_split("dataset1_setting2_test", test_nodes, test_edges, test_edge_types, test_node_types)
        
        print("Done! Full raw directory splits generated.")
