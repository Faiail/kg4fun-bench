import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import torch
import streamlit as st

from src.utils import load_ruamel
import src.experiment as exp_pkg
from src.data.utils import BatchKeys
from src.data.text.template_keys import TemplateKeys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def infer_run_class_name(task: str, architecture: str) -> str:
    """Infers the experiment run class based on task and architecture."""
    if task == "complete":
        return "SharedTextCompleteRun"
    if architecture == "dedicated":
        return "DedicatedTextSingleItemRun"
    return "SharedTextSingleItemRun"


def get_config_path(
    task: str,
    dataset: str,
    split_mode: str,
    architecture: str,
    model: str,
) -> Path:
    """Returns the expected experiment YAML config path."""
    filename = f"{architecture}_{model}.yaml"
    return ROOT_DIR / "configs" / "experiment" / "text" / task / dataset / split_mode / filename


def resolve_checkpoint_path(run) -> Optional[str]:
    """Resolves checkpoint state dict path using run.early_stop.path with directory fallback."""
    ckpt_path = getattr(run.early_stop, "path", None)
    if not ckpt_path:
        return None
    if os.path.exists(ckpt_path):
        return ckpt_path
    # Fallback if checkpoint was saved under models/text/text/...
    alt_path = ckpt_path.replace("models/text/", "models/text/text/")
    if os.path.exists(alt_path):
        return alt_path
    return None


@st.cache_resource(show_spinner="Initializing experiment run and neural model...")
def load_and_init_run(
    config_path_str: str,
    run_class_name: str,
    device: str = "cpu",
    config_mtime: float = 0.0,
    checkpoint_mtime: float = 0.0,
):
    """Loads YAML config, instantiates and inits runner, and loads model weights from early_stop.path."""
    if not os.path.exists(config_path_str):
        raise FileNotFoundError(f"Configuration file not found at: {config_path_str}")

    cfg = load_ruamel(config_path_str)

    cfg.setdefault("general", {})
    cfg["general"]["device"] = device
    cfg["general"]["pbar"] = False
    cfg.setdefault("loader", {})
    cfg["loader"]["num_workers"] = 0

    run_cls = getattr(exp_pkg, run_class_name, None)
    if run_cls is None:
        raise ValueError(f"Run class '{run_class_name}' not found in src.experiment.")

    run = run_cls(cfg)
    run.init()

    # Load state dict from run.early_stop.path
    ckpt_path = resolve_checkpoint_path(run)
    if ckpt_path and os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=device)
        run.model.load_state_dict(state_dict)
        run.checkpoint_loaded_from = ckpt_path
    else:
        run.checkpoint_loaded_from = None

    run.device = device
    run.model = run.model.to(device)
    run.model.eval()

    # Initialize per-run embeddings cache
    run._schema_embeddings_cache = {}

    return run


def get_schema_embeddings(run, dataset, task: str, split: str = "val", force_recompute: bool = False):
    """Encodes and caches schema candidate embeddings directly on the runner instance for the given split.
    Guarantees embeddings are recomputed whenever the run or split changes.
    """
    if not hasattr(run, "_schema_embeddings_cache") or force_recompute:
        run._schema_embeddings_cache = {}

    if split not in run._schema_embeddings_cache or force_recompute:
        run.model.eval()
        with torch.no_grad():
            if task == "complete":
                node_loader, edge_loader = dataset.get_schema_items(batch_size=64, num_workers=0)
                embs = run.get_schema_embeddings(node_loader=node_loader, edge_loader=edge_loader)
            else:
                schema_items = dataset.get_schema_items(batch_size=64, num_workers=0)
                embs = run.get_schema_embeddings(schema_items)
            run._schema_embeddings_cache[split] = embs

    return run._schema_embeddings_cache[split]


def find_sample_index(dataset, sample_identifier: Union[int, str], task: str = "node") -> Optional[int]:
    """Finds integer index of sample by index or QID / triple string."""
    total_len = len(dataset)
    if isinstance(sample_identifier, int):
        if 0 <= sample_identifier < total_len:
            return sample_identifier
        return None

    # String search: QID or ID
    sample_identifier = str(sample_identifier).strip()
    if sample_identifier.isdigit():
        idx = int(sample_identifier)
        if 0 <= idx < total_len:
            return idx

    if task == "node" and hasattr(dataset, "dataset"):
        for i, item in enumerate(dataset.dataset):
            if item.get("qid") == sample_identifier:
                return i

    elif task in ["edge", "complete"] and hasattr(dataset, "partitions"):
        for i, item in enumerate(dataset.partitions):
            head = item.get("head_qid", "")
            tail = item.get("tail_qid", "")
            pid = item.get("edge_type", {}).get("pid", "")
            if sample_identifier in (head, tail, pid, f"{head}-{pid}-{tail}"):
                return i

    return None


def get_schema_string_for_node(dataset, kb_id: int) -> str:
    """Gathers human-readable schema string from node dataset."""
    if hasattr(dataset, "get_schema_str"):
        return dataset.get_schema_str(kb_id)
    if kb_id == -1:
        return f"{TemplateKeys.LABEL} Prune Node | {TemplateKeys.DESC} A node which is out of domain and must be pruned."
    info = dataset.schema_info.get(kb_id, {})
    return f"{TemplateKeys.LABEL} {info.get(BatchKeys.ITEM_LABEL, 'UnknownLabel')} | {TemplateKeys.DESC} {info.get(BatchKeys.ITEM_DESC, 'UnknownDescription')}"


def get_schema_string_for_edge(dataset, kb_id: int) -> str:
    """Gathers human-readable schema string from edge dataset."""
    if kb_id == -1:
        return f"{TemplateKeys.LABEL} Prune Edge | {TemplateKeys.DESC} An edge which is out of domain and must be pruned."
    info = dataset.schema_edge_info.get(kb_id, {})
    return f"{TemplateKeys.LABEL} {info.get(BatchKeys.ITEM_LABEL, 'UnknownLabel')} | {TemplateKeys.DESC} {info.get(BatchKeys.ITEM_DESC, 'UnknownDescription')}"


@torch.no_grad()
def predict_sample(
    run,
    dataset,
    loader,
    schema_embeddings,
    sample_idx: int,
    task: str = "node",
    top_k: int = 5,
) -> Dict[str, Any]:
    """1. Loads raw item from dataset.
    2. Applies loader collate_fn.
    3. Runs model inference via run.get_scores.
    4. Gathers strings to visualize directly using the dataset class.
    """
    run.model.eval()
    raw_item = dataset[sample_idx]
    batch = loader.collate_fn([raw_item])

    if task in ["node", "edge"]:
        input_text = batch[BatchKeys.INPUT]
        input_tok = run.tokenize(input_text)
        scores = run.get_scores(input_tok, schema_embeddings, chunk_size=64)
        scores_1d = scores[0].cpu()
        probs_1d = torch.softmax(scores_1d, dim=-1)

        # Predicted class
        pred_idx = scores_1d.argmax(dim=-1).item()
        pred_kb_id = dataset.idx2cls_kb[pred_idx]

        # Ground truth class
        gt_item = batch[BatchKeys.GT]
        gt_idx = gt_item.item() if isinstance(gt_item, torch.Tensor) else gt_item
        gt_kb_id = dataset.idx2cls_kb[gt_idx]

        # Gather strings using dataset class
        if task == "node":
            pred_str = get_schema_string_for_node(dataset, pred_kb_id)
            gt_str = get_schema_string_for_node(dataset, gt_kb_id)
            sample_id = raw_item.get(BatchKeys.QID, f"Node #{sample_idx}")
        else:
            pred_str = get_schema_string_for_edge(dataset, pred_kb_id)
            gt_str = get_schema_string_for_edge(dataset, gt_kb_id)
            raw_part = dataset.partitions[sample_idx]
            sample_id = f"{raw_part.get('head_qid')} ──({raw_part.get('edge_type', {}).get('pid')})──> {raw_part.get('tail_qid')}"

        # Top-K candidate breakdown
        k = min(top_k, scores_1d.size(0))
        top_scores, top_indices = torch.topk(scores_1d, k=k)
        top_probs = probs_1d[top_indices]

        candidates = []
        for i in range(k):
            c_idx = top_indices[i].item()
            c_kb_id = dataset.idx2cls_kb[c_idx]
            c_str = get_schema_string_for_node(dataset, c_kb_id) if task == "node" else get_schema_string_for_edge(dataset, c_kb_id)
            candidates.append({
                "rank": i + 1,
                "kb_id": c_kb_id,
                "schema_str": c_str,
                "score": float(top_scores[i].item()),
                "prob": float(top_probs[i].item()),
                "is_match": (c_kb_id == gt_kb_id),
            })

        return {
            "task": task,
            "sample_idx": sample_idx,
            "sample_id": sample_id,
            "input_string": input_text[0] if isinstance(input_text, list) else str(input_text),
            "pred_kb_id": pred_kb_id,
            "pred_str": pred_str,
            "gt_kb_id": gt_kb_id,
            "gt_str": gt_str,
            "is_match": (pred_kb_id == gt_kb_id),
            "candidates": candidates,
            "raw_item": raw_item,
        }

    elif task == "complete":
        schema_node_embs, schema_edge_embs = schema_embeddings
        head_tok = run.tokenize(batch[BatchKeys.INPUT_HEAD])
        tail_tok = run.tokenize(batch[BatchKeys.INPUT_TAIL])
        rel_tok = run.tokenize(batch[BatchKeys.INPUT_REL])

        head_scores = run.get_scores(head_tok, schema_node_embs, chunk_size=64)[0].cpu()
        tail_scores = run.get_scores(tail_tok, schema_node_embs, chunk_size=64)[0].cpu()
        rel_scores = run.get_scores(rel_tok, schema_edge_embs, chunk_size=64)[0].cpu()

        head_pred_id = dataset.node_idx2cls_kb[head_scores.argmax().item()]
        tail_pred_id = dataset.node_idx2cls_kb[tail_scores.argmax().item()]
        rel_pred_id = dataset.rel_idx2cls_kb[rel_scores.argmax().item()]

        gt = batch[BatchKeys.GT]
        head_gt_id = dataset.node_idx2cls_kb[gt[BatchKeys.HEAD].item()]
        tail_gt_id = dataset.node_idx2cls_kb[gt[BatchKeys.TAIL].item()]
        rel_gt_id = dataset.rel_idx2cls_kb[gt[BatchKeys.REL].item()]

        head_pred_str = get_schema_string_for_node(dataset, head_pred_id)
        head_gt_str = get_schema_string_for_node(dataset, head_gt_id)
        tail_pred_str = get_schema_string_for_node(dataset, tail_pred_id)
        tail_gt_str = get_schema_string_for_node(dataset, tail_gt_id)
        rel_pred_str = get_schema_string_for_edge(dataset, rel_pred_id)
        rel_gt_str = get_schema_string_for_edge(dataset, rel_gt_id)

        sample_id = f"{batch[BatchKeys.HEAD_QID][0]} ──({batch[BatchKeys.PID][0]})──> {batch[BatchKeys.TAIL_QID][0]}"

        return {
            "task": "complete",
            "sample_idx": sample_idx,
            "sample_id": sample_id,
            "input_string": f"Head: {batch[BatchKeys.INPUT_HEAD][0]} | Tail: {batch[BatchKeys.INPUT_TAIL][0]} | Rel: {batch[BatchKeys.INPUT_REL][0]}",
            "complete_preds": {
                "head": {"pred_id": head_pred_id, "pred_str": head_pred_str, "gt_id": head_gt_id, "gt_str": head_gt_str, "match": head_pred_id == head_gt_id},
                "tail": {"pred_id": tail_pred_id, "pred_str": tail_pred_str, "gt_id": tail_gt_id, "gt_str": tail_gt_str, "match": tail_pred_id == tail_gt_id},
                "rel": {"pred_id": rel_pred_id, "pred_str": rel_pred_str, "gt_id": rel_gt_id, "gt_str": rel_gt_str, "match": rel_pred_id == rel_gt_id},
            },
            "is_match": (head_pred_id == head_gt_id and tail_pred_id == tail_gt_id and rel_pred_id == rel_gt_id),
            "raw_item": raw_item,
        }

    raise ValueError(f"Unsupported task: {task}")
