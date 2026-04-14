"""
DrawNet Evaluation
------------------
Metric functions for both tasks:

  Intent    : Top-1 accuracy, Top-5 accuracy, macro F1
  Deviation : Per-class AUROC, macro F1, Hamming loss, subset accuracy
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, hamming_loss


def evaluate_intent(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """
    Parameters
    ----------
    logits : (N, C)
    labels : (N,) integer class indices

    Returns
    -------
    dict with top1_acc, top5_acc, macro_f1
    """
    preds_top1 = logits.argmax(dim=1).cpu().numpy()
    labels_np  = labels.cpu().numpy()

    top1     = accuracy_score(labels_np, preds_top1)
    k        = min(5, logits.shape[1])
    topk     = logits.topk(k, dim=1).indices.cpu().numpy()
    top5     = float(np.mean([labels_np[i] in topk[i] for i in range(len(labels_np))]))
    macro_f1 = f1_score(labels_np, preds_top1, average="macro", zero_division=0)

    return {"top1_acc": top1, "top5_acc": top5, "macro_f1": macro_f1}


def evaluate_deviation(
    logits:    torch.Tensor,
    labels:    torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """
    Parameters
    ----------
    logits : (N, 4) raw scores
    labels : (N, 4) binary ground truth

    Returns
    -------
    dict with hamming_loss, subset_acc, macro_f1, per_class_auroc
    """
    probs     = torch.sigmoid(logits).cpu().numpy()
    preds     = (probs >= threshold).astype(int)
    labels_np = labels.cpu().numpy()

    h_loss     = hamming_loss(labels_np, preds)
    subset_acc = accuracy_score(labels_np, preds)
    macro_f1   = f1_score(labels_np, preds, average="macro", zero_division=0)

    per_class_auroc = []
    for c in range(labels_np.shape[1]):
        if labels_np[:, c].sum() > 0:
            per_class_auroc.append(float(roc_auc_score(labels_np[:, c], probs[:, c])))
        else:
            per_class_auroc.append(float("nan"))

    return {
        "hamming_loss":    h_loss,
        "subset_acc":      subset_acc,
        "macro_f1":        macro_f1,
        "per_class_auroc": per_class_auroc,
    }
