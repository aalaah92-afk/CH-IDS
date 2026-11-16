import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef


def binary_metrics(y, pred, score=None):
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    acc = (tp + tn) / len(y)
    prec = tp / (tp + fp) if tp + fp else 0.0
    dr = tp / (tp + fn) if tp + fn else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    f1 = 2 * prec * dr / (prec + dr) if (prec + dr) else 0.0
    out = dict(Acc=acc, Prec=prec, DR=dr, FPR=fpr, F1=f1, MCC=matthews_corrcoef(y, pred))
    if score is not None and len(np.unique(y)) == 2:
        out["ROC_AUC"] = roc_auc_score(y, score)
        out["PR_AUC"] = average_precision_score(y, score)
    return out
