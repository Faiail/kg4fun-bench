import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import streamlit as st
from src.utils import load_json
from .config import get_data_paths, find_prediction_file, find_metrics_file


@st.cache_data(show_spinner="Loading schema information...")
def load_schema_info(schema_path_str: Optional[str]) -> Dict[int, Dict[str, Any]]:
    """Loads schema type information and indexes it by integer class/relation index."""
    if not schema_path_str or not os.path.exists(schema_path_str):
        return {}
    raw_list = load_json(schema_path_str)
    schema_map = {}
    for item in raw_list:
        idx = item.get("idx")
        if idx is not None:
            schema_map[int(idx)] = {
                "idx": int(idx),
                "label": item.get("itemLabel", "Unknown Schema Label"),
                "description": item.get("itemDescription", "No description provided"),
            }
    return schema_map


def format_schema_entry(schema_map: Dict[int, Dict[str, Any]], idx: Optional[int], task: str = "node") -> Dict[str, Any]:
    """Formats a schema class index into a human-readable label, description, and full string."""
    if idx is None:
        return {
            "idx": None,
            "label": "No Prediction Recorded",
            "description": "Model prediction not found for this sample.",
            "is_prune": False,
            "formatted_string": "No Prediction",
        }

    idx = int(idx)
    if idx == -1:
        prune_label = "Prune Node" if task == "node" else "Prune Edge"
        prune_desc = (
            "A node which is out of domain and must be pruned."
            if task == "node"
            else "An edge which is out of domain and must be pruned."
        )
        return {
            "idx": -1,
            "label": prune_label,
            "description": prune_desc,
            "is_prune": True,
            "formatted_string": f"[LABEL] {prune_label} | [DESC] {prune_desc}",
        }

    entry = schema_map.get(idx)
    if entry:
        label = entry.get("label", f"Class {idx}")
        desc = entry.get("description", "No description available")
        return {
            "idx": idx,
            "label": label,
            "description": desc,
            "is_prune": False,
            "formatted_string": f"[LABEL] {label} | [DESC] {desc}",
        }

    return {
        "idx": idx,
        "label": f"Unknown Class #{idx}",
        "description": "Schema entry not found in dictionary.",
        "is_prune": False,
        "formatted_string": f"[LABEL] Class {idx}",
    }


@st.cache_data(show_spinner="Loading node dataset...")
def load_node_dataset(dataset: str, split_mode: str, split: str) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """Loads nodes, node metadata, and node schema for the chosen split."""
    paths = get_data_paths(dataset, split_mode, split)
    nodes_path = paths["nodes"]
    node_info_path = paths["node_info"]
    schema_path = paths["node_schema"]

    if not nodes_path.exists():
        return [], {}

    raw_nodes = load_json(str(nodes_path))
    node_info = load_json(str(node_info_path)) if node_info_path.exists() else {}
    schema_map = load_schema_info(str(schema_path) if schema_path else None)

    samples = []
    for idx, node in enumerate(raw_nodes):
        qid = node["qid"]
        info = node_info.get(qid, {})
        label = info.get("itemLabel") or node.get("itemLabel") or "Unknown"
        desc = info.get("itemDescription") or "No description available"
        gt_idx = int(node["cls_idx"])

        samples.append({
            "index": idx,
            "sample_id": qid,
            "qid": qid,
            "label": label,
            "description": desc,
            "gt_idx": gt_idx,
            "full_input_text": f"[LABEL] {label} | [DESC] {desc}",
        })

    return samples, schema_map


@st.cache_data(show_spinner="Loading edge dataset...")
def load_edge_dataset(
    dataset: str, split_mode: str, split: str, max_partitions: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """Loads edge partitions, edge mapping, entity metadata, and edge schema."""
    paths = get_data_paths(dataset, split_mode, split)
    partition_dir = paths["edge_partitions"]
    mapping_path = paths["edge_mapping"]
    node_info_path = paths["node_info"]
    edge_info_path = paths["edge_info"]
    schema_path = paths["edge_schema"]

    if not partition_dir.exists() or not mapping_path.exists():
        return [], {}

    node_info = load_json(str(node_info_path)) if node_info_path.exists() else {}
    input_edge_info = load_json(str(edge_info_path)) if edge_info_path.exists() else {}
    schema_map = load_schema_info(str(schema_path) if schema_path else None)

    raw_mapping = load_json(str(mapping_path))
    mapping = {tuple(x["edge_type"]): x["component_id"] for x in raw_mapping}

    partition_files = sorted(os.listdir(partition_dir))
    if max_partitions:
        partition_files = partition_files[:max_partitions]

    samples = []
    sample_counter = 0

    for fname in partition_files:
        fpath = partition_dir / fname
        part_data = load_json(str(fpath))
        for raw_item in part_data:
            head_qid = raw_item.get("head_qid", "")
            tail_qid = raw_item.get("tail_qid", "")
            edge_type = raw_item.get("edge_type", {})
            head_cls = edge_type.get("head_cls", -1)
            tail_cls = edge_type.get("tail_cls", -1)
            pid = edge_type.get("pid", "")

            head_info = node_info.get(head_qid, {})
            tail_info = node_info.get(tail_qid, {})
            rel_info = input_edge_info.get(pid, {})

            head_label = head_info.get("itemLabel", "Unknown")
            head_desc = head_info.get("itemDescription", "No description")
            tail_label = tail_info.get("itemLabel", "Unknown")
            tail_desc = tail_info.get("itemDescription", "No description")
            rel_label = rel_info.get("itemLabel", edge_type.get("label", pid))
            rel_desc = rel_info.get("itemDescription", "No description")

            if head_cls == -1 or tail_cls == -1:
                gt_idx = -1
            else:
                gt_idx = mapping.get((head_cls, pid, tail_cls), -1)

            input_str = (
                f"[HEAD] [LABEL] {head_label} | [DESC] {head_desc} | "
                f"[TAIL] [LABEL] {tail_label} | [DESC] {tail_desc} | "
                f"[PID] [LABEL] {rel_label} | [DESC] {rel_desc}"
            )

            samples.append({
                "index": sample_counter,
                "sample_id": f"{head_qid} ──({pid}: {rel_label})──> {tail_qid}",
                "head_qid": head_qid,
                "head_label": head_label,
                "head_desc": head_desc,
                "tail_qid": tail_qid,
                "tail_label": tail_label,
                "tail_desc": tail_desc,
                "pid": pid,
                "rel_label": rel_label,
                "rel_desc": rel_desc,
                "head_cls": head_cls,
                "tail_cls": tail_cls,
                "gt_idx": gt_idx,
                "full_input_text": input_str,
            })
            sample_counter += 1

    return samples, schema_map


@st.cache_data(show_spinner="Loading model predictions...")
def load_predictions(prediction_path_str: Optional[str]) -> Tuple[Optional[Dict[Any, Any]], Optional[List[Any]]]:
    """Loads the model predictions. Returns a dict mapping key -> predicted_idx and raw list."""
    if not prediction_path_str or not os.path.exists(prediction_path_str):
        return None, None

    raw_preds = load_json(prediction_path_str)
    lookup_map = {}

    if isinstance(raw_preds, list):
        for idx, item in enumerate(raw_preds):
            if isinstance(item, (list, tuple)) and len(item) == 2:
                key, pred_cls = item[0], item[1]
                lookup_map[key] = int(pred_cls)
            elif isinstance(item, dict):
                lookup_map[item.get("qid", idx)] = int(item.get("pred", -1))
            else:
                lookup_map[idx] = int(item)

    return lookup_map, raw_preds


@st.cache_data(show_spinner="Loading evaluation metrics...")
def load_metrics(metrics_path_str: Optional[str]) -> Optional[Dict[str, Any]]:
    """Loads metrics summary if present."""
    if not metrics_path_str or not os.path.exists(metrics_path_str):
        return None
    return load_json(metrics_path_str)
