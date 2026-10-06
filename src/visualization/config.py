import os
from pathlib import Path
from typing import Dict, Any, Optional

DATASETS = ["dataset1", "dataset2"]
TASKS = ["node", "edge", "complete"]
SPLIT_MODES = ["hybrid", "inductive"]
ARCHITECTURES = ["shared", "dedicated"]
MODELS = ["bert", "bge", "roberta", "modernbert"]
DATA_SPLITS = ["test", "val", "train"]

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def get_model_dir(
    task: str,
    architecture: str,
    model: str,
    dataset: str,
    split_mode: str,
) -> Path:
    """Returns the expected model directory where checkpoints and predictions are saved."""
    model_name = f"{architecture}_{model}"
    # Standard location: models/text/{task}/{model_name}/{dataset}/{split_mode}
    standard_path = ROOT_DIR / "models" / "text" / task / model_name / dataset / split_mode
    if standard_path.exists():
        return standard_path

    # Fallback to alternate directory structure if text/text exists
    alt_path = ROOT_DIR / "models" / "text" / "text" / task / model_name / dataset / split_mode
    if alt_path.exists():
        return alt_path

    return standard_path


def find_prediction_file(model_dir: Path) -> Optional[Path]:
    """Finds inferece.json or inference.json in the model directory."""
    if not model_dir.exists():
        return None
    for name in ["inferece.json", "inference.json", "predictions.json"]:
        candidate = model_dir / name
        if candidate.exists():
            return candidate
    return None


def find_metrics_file(model_dir: Path) -> Optional[Path]:
    """Finds test_metrics.json in the model directory."""
    if not model_dir.exists():
        return None
    for name in ["test_metrics.json", "metrics.json"]:
        candidate = model_dir / name
        if candidate.exists():
            return candidate
    return None


def get_data_paths(dataset: str, split_mode: str, split: str) -> Dict[str, Path]:
    """Resolves data paths for the specified dataset, split mode, and split."""
    base_split_dir = ROOT_DIR / "data" / "raw" / "kg4fun" / dataset / "splits" / split_mode / split

    node_schema_dir = base_split_dir / "node_types" / "summarization"
    node_schema_path = None
    if node_schema_dir.exists():
        for sub in ["qwen3-8b", "qwen-8b"]:
            p = node_schema_dir / sub / "node_type_info.json"
            if p.exists():
                node_schema_path = p
                break
    if node_schema_path is None:
        fallback = base_split_dir / "node_type_info.json"
        if fallback.exists():
            node_schema_path = fallback

    edge_schema_dir = base_split_dir / "edge_types" / "summarization"
    edge_schema_path = None
    if edge_schema_dir.exists():
        for sub in ["qwen-8b", "qwen3-8b"]:
            p = edge_schema_dir / sub / "edge_type_info.json"
            if p.exists():
                edge_schema_path = p
                break

    return {
        "nodes": base_split_dir / "nodes.json",
        "node_info": base_split_dir / "node_info.json",
        "node_schema": node_schema_path,
        "edge_partitions": base_split_dir / "edges" / "partition",
        "edge_mapping": base_split_dir / "edges" / "edge_component_mapping.json",
        "edge_info": base_split_dir / "edges" / "edge_info.json",
        "edge_schema": edge_schema_path,
    }
