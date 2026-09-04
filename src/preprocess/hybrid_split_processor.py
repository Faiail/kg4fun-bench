import os
import random
import shutil
import networkx as nx
from tqdm import tqdm

from src.utils import load_json, save_json
from src.utils import ParameterKeys
from src.preprocess.ontomap.kg4fun_fields import ClassFields, RelFields

class HybridSplitProcessor:
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
            ParameterKeys.OUT_DIR, f"data/processed/kg4fun/hybrid_splits"
        )
        os.makedirs(self.out_dir, exist_ok=True)
        self.pbar = general_parameters.get(ParameterKeys.PBAR, True)
        self.train_ratio = general_parameters.get(ParameterKeys.TRAIN_RATIO, 0.7)
        self.val_ratio = general_parameters.get(ParameterKeys.VAL_RATIO, 0.1)

    def _init_data(self):
        dataset_parameters = self.parameters.get(ParameterKeys.DATA, dict())
        self.data_dir = dataset_parameters.get(ParameterKeys.DATA_DIR, "./")

        self.global_node_types = load_json(os.path.join(self.data_dir, f"{ParameterKeys.NODE_TYPES}.json"))
        self.global_edge_types = load_json(os.path.join(self.data_dir, ParameterKeys.EDGES, f"{ParameterKeys.EDGE_TYPES}.json"))
        
        print("Loading nodes...")
        self.nodes = load_json(os.path.join(self.data_dir, f"{ParameterKeys.NODES}.json"))
        
        print("Loading edges from partitions...")
        import glob
        edge_files = glob.glob(
            os.path.join(self.data_dir, ParameterKeys.EDGES, ParameterKeys.PARTITION, "part_*.json")
        )
        self.all_edges = []
        iter_files = tqdm(edge_files, desc="Loading edge partitions") if self.pbar else edge_files
        for ef in iter_files:
            self.all_edges.extend(load_json(ef))
            
        print(f"Total edges loaded: {len(self.all_edges)}")

    def _grow_island(self, G, target_size, unassigned, valid_subset=None):
        import collections
        island = set()
        candidates = set(unassigned) if valid_subset is None else unassigned.intersection(valid_subset)
        if not candidates:
            return island
            
        while len(island) < target_size and candidates:
            # Pop an arbitrary element efficiently from the set
            start_node = next(iter(candidates))
            queue = collections.deque([start_node])
            unassigned.remove(start_node)
            candidates.remove(start_node)
            island.add(start_node)
            
            while queue and len(island) < target_size:
                node = queue.popleft()
                neighbors = [n for n in G.neighbors(node) if n in candidates]
                random.shuffle(neighbors)
                
                for neighbor in neighbors:
                    if len(island) >= target_size:
                        break
                    unassigned.remove(neighbor)
                    candidates.remove(neighbor)
                    island.add(neighbor)
                    queue.append(neighbor)
        return island

    def filter_schema(self, target_node_types):
        filtered_edge_types = [
            et for et in self.global_edge_types 
            if et[ParameterKeys.HEAD_CLS] in target_node_types and et[ParameterKeys.TAIL_CLS] in target_node_types
        ]
        
        filtered_node_types = {
            qid: info for qid, info in self.global_node_types.items() 
            if info[ParameterKeys.CLS_IDX] in target_node_types
        }
        
        return {
            ParameterKeys.EDGE_TYPES: filtered_edge_types,
            ParameterKeys.NODE_TYPES: filtered_node_types
        }

    def _export_split(self, split_name, target_schema_nodes, target_instance_qids):
        split_dir = os.path.join(self.out_dir, split_name)
        edges_dir = os.path.join(split_dir, ParameterKeys.EDGES)
        partition_dir = os.path.join(edges_dir, ParameterKeys.PARTITION)
        os.makedirs(partition_dir, exist_ok=True)
        
        # 1. Schema
        # If target_schema_nodes is None, it means we use the global schema (for val/test)
        if target_schema_nodes is None:
            schema = {
                ParameterKeys.EDGE_TYPES: self.global_edge_types,
                ParameterKeys.NODE_TYPES: self.global_node_types
            }
            target_schema_nodes = set(info[ParameterKeys.CLS_IDX] for info in self.global_node_types.values())
        else:
            schema = self.filter_schema(target_schema_nodes)
            
        valid_pids = {et[ParameterKeys.PID] for et in schema[ParameterKeys.EDGE_TYPES]}
        
        # 2. Filter input graph
        nodes_subset = [n for n in self.nodes if n[ParameterKeys.QID] in target_instance_qids]
        
        edges_subset = [
            e for e in self.all_edges 
            if e[ParameterKeys.HEAD_QID] in target_instance_qids and e[ParameterKeys.TAIL_QID] in target_instance_qids
               and e[ParameterKeys.EDGE_TYPE][ParameterKeys.HEAD_CLS] in target_schema_nodes
               and e[ParameterKeys.EDGE_TYPE][ParameterKeys.TAIL_CLS] in target_schema_nodes
        ]
        
        # Save schemas
        save_json(schema[ParameterKeys.NODE_TYPES], os.path.join(split_dir, f"{ParameterKeys.NODE_TYPES}.json"))
        save_json(schema[ParameterKeys.EDGE_TYPES], os.path.join(edges_dir, f"{ParameterKeys.EDGE_TYPES}.json"))
        
        # Save nodes and edges
        save_json(nodes_subset, os.path.join(split_dir, f"{ParameterKeys.NODES}.json"))
        save_json(edges_subset, os.path.join(partition_dir, f"{ParameterKeys.PART_0}.json"))
        
        # 3. Filter other information files
        src_nti = os.path.join(self.data_dir, f"{ParameterKeys.NODE_TYPE_INFO}.json")
        if os.path.exists(src_nti):
            nti = load_json(src_nti)
            filtered_nti = {k: v for k, v in nti.items() if k in schema[ParameterKeys.NODE_TYPES]}
            save_json(filtered_nti, os.path.join(split_dir, f"{ParameterKeys.NODE_TYPE_INFO}.json"))

        src_ni = os.path.join(self.data_dir, f"{ParameterKeys.NODE_INFO}.json")
        if os.path.exists(src_ni):
            ni = load_json(src_ni)
            filtered_ni = {k: v for k, v in ni.items() if k in target_instance_qids}
            save_json(filtered_ni, os.path.join(split_dir, f"{ParameterKeys.NODE_INFO}.json"))
        
        src_edge_info = os.path.join(self.data_dir, ParameterKeys.EDGES, f"{ParameterKeys.EDGE_INFO}.json")
        if os.path.exists(src_edge_info):
            ei = load_json(src_edge_info)
            filtered_ei = {k: v for k, v in ei.items() if k in valid_pids}
            save_json(filtered_ei, os.path.join(edges_dir, f"{ParameterKeys.EDGE_INFO}.json"))

        # Filter node_types summarizations
        src_nt_dir = os.path.join(self.data_dir, ParameterKeys.NODE_TYPES)
        if os.path.isdir(src_nt_dir):
            shutil.copytree(src_nt_dir, os.path.join(split_dir, ParameterKeys.NODE_TYPES), dirs_exist_ok=True)
            for root, _, files in os.walk(os.path.join(split_dir, ParameterKeys.NODE_TYPES)):
                for file in files:
                    if file.endswith(".json"):
                        path = os.path.join(root, file)
                        try:
                            data = load_json(path)
                            if isinstance(data, list) and len(data) > 0 and ClassFields.IDX in data[0]:
                                filtered_data = [d for d in data if d[ClassFields.IDX] in target_schema_nodes]
                                save_json(filtered_data, path)
                        except Exception:
                            pass

        # Filter edge_types summarizations
        src_et_dir = os.path.join(self.data_dir, ParameterKeys.EDGE_TYPES)
        if os.path.isdir(src_et_dir):
            shutil.copytree(src_et_dir, os.path.join(split_dir, ParameterKeys.EDGE_TYPES), dirs_exist_ok=True)
            for root, _, files in os.walk(os.path.join(split_dir, ParameterKeys.EDGE_TYPES)):
                for file in files:
                    if file.endswith(".json"):
                        path = os.path.join(root, file)
                        try:
                            data = load_json(path)
                            if isinstance(data, list) and len(data) > 0 and RelFields.PID in data[0]:
                                filtered_data = [d for d in data if d[RelFields.PID] in valid_pids]
                                save_json(filtered_data, path)
                        except Exception:
                            pass
        
        # Connected components and mappings
        src_cc = os.path.join(self.data_dir, ParameterKeys.EDGES, f"{ParameterKeys.CONNECTED_COMPONENTS}.json")
        if os.path.exists(src_cc):
            cc = load_json(src_cc)
            filtered_cc = []
            for comp in cc:
                valid_ets = [et for et in comp.get(ParameterKeys.EDGE_TYPES, []) if et[0] in valid_pids]
                if valid_ets:
                    comp_copy = comp.copy()
                    comp_copy[ParameterKeys.EDGE_TYPES] = valid_ets
                    filtered_cc.append(comp_copy)
            save_json(filtered_cc, os.path.join(edges_dir, f"{ParameterKeys.CONNECTED_COMPONENTS}.json"))
            
        src_mapping = os.path.join(self.data_dir, ParameterKeys.EDGES, f"{ParameterKeys.EDGE_COMPONENT_MAPPING}.json")
        if os.path.exists(src_mapping):
            mapping = load_json(src_mapping)
            filtered_mapping = [m for m in mapping if m[ParameterKeys.EDGE_TYPE][0] in valid_pids]
            save_json(filtered_mapping, os.path.join(edges_dir, f"{ParameterKeys.EDGE_COMPONENT_MAPPING}.json"))


    def __call__(self):
        print("Building undirected SCHEMA graph for partitioning...")
        schema_graph = nx.Graph()
        for qid, info in self.global_node_types.items():
            schema_graph.add_node(info[ParameterKeys.CLS_IDX])
        for et in self.global_edge_types:
            schema_graph.add_edge(et[ParameterKeys.HEAD_CLS], et[ParameterKeys.TAIL_CLS])
            
        total_schema_nodes = len(schema_graph.nodes())
        print(f"Total schema nodes: {total_schema_nodes}")
        
        unassigned_schema = set(schema_graph.nodes())
        train_schema_target = int(self.train_ratio * total_schema_nodes)
        
        print("Assigning SCHEMA nodes to train island...")
        train_schema_nodes = self._grow_island(schema_graph, train_schema_target, unassigned_schema)
        # We don't need to partition the rest of the schema, because val/test use the WHOLE schema.

        print("Building undirected INSTANCE graph for partitioning...")
        instance_graph = nx.Graph()
        for n in self.nodes:
            instance_graph.add_node(n[ParameterKeys.QID], cls_idx=n[ParameterKeys.CLS_IDX])
        
        iter_edges2 = tqdm(self.all_edges, desc="Adding edges to instance graph") if self.pbar else self.all_edges
        for e in iter_edges2:
            instance_graph.add_edge(e[ParameterKeys.HEAD_QID], e[ParameterKeys.TAIL_QID])
            
        total_instance_nodes = len(instance_graph.nodes())
        print(f"Total instance nodes: {total_instance_nodes}")
        
        # Partition instance graph
        # Train instances must ONLY be selected from base instances (those whose class is in train_schema_nodes)
        base_instances = {n for n, attr in instance_graph.nodes(data=True) if attr["cls_idx"] in train_schema_nodes}
        
        unassigned_instances = set(instance_graph.nodes())
        train_instance_target = int(self.train_ratio * total_instance_nodes)
        val_instance_target = int(self.val_ratio * total_instance_nodes)
        
        print("Growing TRAIN instance island (restricted to train schema classes)...")
        # Ensure we only pick from base_instances for Train
        train_instance_qids = self._grow_island(instance_graph, train_instance_target, unassigned_instances, valid_subset=base_instances)
        
        print("Growing VAL instance island (can include remaining base + new instances)...")
        # Val can pick from ANY remaining unassigned instances
        val_instance_qids = self._grow_island(instance_graph, val_instance_target, unassigned_instances)
        
        print("Assigning remaining to TEST instance island...")
        test_instance_qids = unassigned_instances
        
        print(f"Instance Nodes assigned - Train: {len(train_instance_qids)}, Val: {len(val_instance_qids)}, Test: {len(test_instance_qids)}")
        
        print("Exporting train split (subset schema)...")
        self._export_split("train", train_schema_nodes, train_instance_qids)
        
        print("Exporting val split (global schema)...")
        self._export_split("val", None, val_instance_qids)
        
        print("Exporting test split (global schema)...")
        self._export_split("test", None, test_instance_qids)
        
        print("Done! Hybrid splits generated.")
