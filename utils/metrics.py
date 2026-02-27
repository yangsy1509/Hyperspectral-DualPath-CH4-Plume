import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    roc_curve,
    auc,
    average_precision_score,
)


def binary_metrics_from_probs(y_true, y_prob, thr=0.5):
    y_true = y_true.reshape(-1)
    y_prob = y_prob.reshape(-1)
    y_true = (y_true >= 1).astype(np.uint8)
    y_pred = (y_prob >= thr).astype(np.uint8)

    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "iou": jaccard_score(y_true, y_pred, zero_division=0),
    }
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        out["roc_auc"] = auc(fpr, tpr)
        out["prc_auc"] = average_precision_score(y_true, y_prob)
    else:
        out["roc_auc"] = 0.0
        out["prc_auc"] = 0.0
    return out
