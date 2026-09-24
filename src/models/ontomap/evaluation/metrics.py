# -*- coding: utf-8 -*-
from typing import Dict, List
import torch
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score
from tqdm import tqdm


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

