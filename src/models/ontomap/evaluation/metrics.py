# -*- coding: utf-8 -*-
from typing import Dict, List
import torch
from torchmetrics.classification import (
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)
from tqdm import tqdm


def compute_closed_world_macro_metrics(
    predicts: List,
    references: List,
    confidence_th: float = 0.5,
) -> Dict:
    """
    Computes closed-world multiclass macro-averaged metrics matching TextRun.
    For any source entity with no prediction >= confidence_th, defaults to the prune class ('cls_-1' or 'rel_-1').
    Returns macro precision, recall, and f1 on the [0.0, 1.0] scale.
    """
    node_gt = {}
    edge_gt = {}
    for r in references:
        src = str(r["source"] if isinstance(r, dict) else r[0])
        tgt = str(r["target"] if isinstance(r, dict) else r[1])
        if not (tgt.startswith("cls_") or tgt.startswith("rel_")):
            tgt = f"cls_{tgt}" if src.startswith("Q") else f"rel_{tgt}"

        if src.startswith("Q"):
            node_gt[src] = tgt
        else:
            edge_gt[src] = tgt

    node_preds = {}
    edge_preds = {}
    for p in predicts:
        src = str(p["source"] if isinstance(p, dict) else p[0])
        tgt = str(p["target"] if isinstance(p, dict) else p[1])
        score = float(p.get("score", 1.0) if isinstance(p, dict) else 1.0)

        if not (tgt.startswith("cls_") or tgt.startswith("rel_")):
            tgt = f"cls_{tgt}" if src.startswith("Q") else f"rel_{tgt}"

        target_dict = node_preds if src.startswith("Q") else edge_preds

        if src not in target_dict or score > target_dict[src][1]:
            target_dict[src] = (tgt, score)

    def compute_macro_partition(gt_dict, pred_dict, default_prune_class):
        if not gt_dict:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        all_classes = set(gt_dict.values()) | {default_prune_class}
        for tgt, score in pred_dict.values():
            if score >= confidence_th:
                all_classes.add(tgt)

        class_to_idx = {c: i for i, c in enumerate(sorted(all_classes))}
        num_classes = len(class_to_idx)

        y_true = []
        y_pred = []

        all_sources = sorted(gt_dict.keys())
        for src in all_sources:
            true_cls = gt_dict[src]
            y_true.append(class_to_idx.get(true_cls, class_to_idx[default_prune_class]))

            if src in pred_dict and pred_dict[src][1] >= confidence_th:
                predicted_cls = pred_dict[src][0]
                y_pred.append(class_to_idx.get(predicted_cls, class_to_idx[default_prune_class]))
            else:
                y_pred.append(class_to_idx[default_prune_class])

        pred_t = torch.tensor(y_pred, dtype=torch.long)
        true_t = torch.tensor(y_true, dtype=torch.long)

        p = MulticlassPrecision(num_classes=num_classes, average="macro")(pred_t, true_t).item()
        r = MulticlassRecall(num_classes=num_classes, average="macro")(pred_t, true_t).item()
        f1 = MulticlassF1Score(num_classes=num_classes, average="macro")(pred_t, true_t).item()

        return {"precision": p, "recall": r, "f1": f1}

    node_metrics = compute_macro_partition(node_gt, node_preds, "cls_-1")
    edge_metrics = compute_macro_partition(edge_gt, edge_preds, "rel_-1")

    return {
        "node": node_metrics,
        "rel": edge_metrics,
    }


def evaluation_report(predicts: List, references: List, llm_confidence_th: float = 0.5) -> Dict:
    """
    Evaluates alignment predictions as a binary classification task.
    Supports multi-target mapping by treating each (source, target) pair independently.
    Metrics are computed using torchmetrics and partitioned into Nodes (Classes) and Edges (Relations).

    :param predicts: List of dicts with keys "source", "target", "score"
    :param references: List of ground truth dicts with keys "source", "target" (only positive references expected)
    :param llm_confidence_th: Threshold to classify a prediction as positive (1) or negative (0)
    :return: Dictionary containing node and edge metrics (precision, recall, f1)
    """
    pos_pairs = set((str(r["source"]), str(r["target"])) for r in references)

    evaluated_pairs = {}
    for p in predicts:
        # Default to 1.0 if score is missing but it was predicted (e.g. baseline approaches)
        score = float(p.get("score", 1.0))
        evaluated_pairs[(str(p["source"]), str(p["target"]))] = score

    # Include any ground truth pairs the model completely missed (implicit score of 0)
    for pair in pos_pairs:
        if pair not in evaluated_pairs:
            evaluated_pairs[pair] = 0.0

    node_y_true, node_y_pred = [], []
    edge_y_true, edge_y_pred = [], []

    for (src, tgt), score in evaluated_pairs.items():
        is_pos = 1 if (src, tgt) in pos_pairs else 0
        pred = 1 if score >= llm_confidence_th else 0

        # Partition by node vs edge type. (Nodes typically start with 'Q' in wikidata/kg4fun)
        if src.startswith('Q'):
            node_y_true.append(is_pos)
            node_y_pred.append(pred)
        else:
            edge_y_true.append(is_pos)
            edge_y_pred.append(pred)

    results = {}
    
    # Helper to compute metrics safely
    def compute_partition(y_pred, y_true, prefix):
        if len(y_true) == 0:
            return {f"{prefix}_precision": 0.0, f"{prefix}_recall": 0.0, f"{prefix}_f-score": 0.0, f"{prefix}_count": 0}
            
        pred_t = torch.tensor(y_pred)
        true_t = torch.tensor(y_true)
        
        p = BinaryPrecision()(pred_t, true_t).item() * 100
        r = BinaryRecall()(pred_t, true_t).item() * 100
        f1 = BinaryF1Score()(pred_t, true_t).item() * 100
        
        return {
            f"{prefix}_precision": p,
            f"{prefix}_recall": r,
            f"{prefix}_f-score": f1,
            f"{prefix}_evaluated_pairs": len(y_true)
        }

    results.update(compute_partition(node_y_pred, node_y_true, "node"))
    results.update(compute_partition(edge_y_pred, edge_y_true, "edge"))
    
    # Global metrics for backward compatibility
    global_y_pred = node_y_pred + edge_y_pred
    global_y_true = node_y_true + edge_y_true
    results.update(compute_partition(global_y_pred, global_y_true, "global"))

    return results

