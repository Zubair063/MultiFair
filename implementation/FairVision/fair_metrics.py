
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, auc

def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _validate_binary_labels(labels: np.ndarray):
    uniq = np.unique(labels)
    if not set(uniq).issubset({0, 1}):
        raise ValueError(f"Labels must be binary {0,1}. Got unique labels: {uniq}")


def _safe_group_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    uniq = np.unique(y_true)
    if len(uniq) < 2:
        return None
    return roc_auc_score(y_true, y_prob)

def overall_auc(probs, labels) -> float:
    p = _to_numpy(probs).astype(float)
    y = _to_numpy(labels).astype(int)
    _validate_binary_labels(y)
    return roc_auc_score(y, p)


def group_aucs(probs, labels, demographics) -> Dict[Any, Optional[float]]:
    p = _to_numpy(probs).astype(float)
    y = _to_numpy(labels).astype(int)
    g = _to_numpy(demographics)
    _validate_binary_labels(y)

    results: Dict[Any, Optional[float]] = {}
    for grp in np.unique(g):
        mask = (g == grp)
        auc_val = _safe_group_auc(y[mask], p[mask])
        results[grp.item() if hasattr(grp, "item") else grp] = auc_val
    return results


def equity_scaled_auc(probs, labels, demographics, alpha: float = 1.0) -> float:
    p = _to_numpy(probs).astype(float)
    y = _to_numpy(labels).astype(int)
    g = _to_numpy(demographics)
    _validate_binary_labels(y)

    fpr, tpr, _ = roc_curve(y, p)
    ov = auc(fpr, tpr)

    deviations = []
    for grp in np.unique(g):
        mask = (g == grp)
        auc_g = _safe_group_auc(y[mask], p[mask])
        if auc_g is not None:
            deviations.append(abs(auc_g - ov))
    total_dev = float(np.sum(deviations)) if deviations else 0.0
    return ov / (alpha * total_dev + 1.0)


def demographic_parity_difference(probs, demographics, threshold: float = 0.5) -> float:
    p = _to_numpy(probs).astype(float)
    g = _to_numpy(demographics)
    y_pred = (p >= threshold).astype(int)
    groups = np.unique(g)
    sel_rates = [np.mean(y_pred[g == grp]) for grp in groups]
    diffs = [abs(sel_rates[i] - sel_rates[j]) for i in range(len(sel_rates)) for j in range(i+1, len(sel_rates))]
    return float(np.max(diffs)) if diffs else 0.0


def _tpr_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return tpr, fpr


def equalized_odds_difference(probs, labels, demographics, threshold: float = 0.5) -> Tuple[float, Dict[Any, float], Dict[Any, float]]:
    p = _to_numpy(probs).astype(float)
    y = _to_numpy(labels).astype(int)
    g = _to_numpy(demographics)
    _validate_binary_labels(y)
    y_pred = (p >= threshold).astype(int)
    groups = np.unique(g)
    tpr_by_group, fpr_by_group = {}, {}
    for grp in groups:
        mask = (g == grp)
        tpr, fpr = _tpr_fpr(y[mask], y_pred[mask])
        key = grp.item() if hasattr(grp, "item") else grp
        tpr_by_group[key] = float(tpr)
        fpr_by_group[key] = float(fpr)
    def _max_pairwise_abs(xs):
        diffs = [abs(xs[i] - xs[j]) for i in range(len(xs)) for j in range(i+1, len(xs))]
        return max(diffs) if diffs else 0.0
    tpr_diff = _max_pairwise_abs(list(tpr_by_group.values()))
    fpr_diff = _max_pairwise_abs(list(fpr_by_group.values()))
    return float(max(tpr_diff, fpr_diff)), tpr_by_group, fpr_by_group

def evaluate_all(probs, labels, demographics: Optional[np.ndarray] = None, *, threshold: float = 0.5, alpha: float = 1.0) -> Dict[str, Any]:
    p = _to_numpy(probs).astype(float)
    y = _to_numpy(labels).astype(int)
    _validate_binary_labels(y)
    results: Dict[str, Any] = {
        "overall_auc": overall_auc(p, y),
        "group_aucs": {},
        "es_auc": None,
        "dpd": None,
        "deodds": None,
        "tpr_by_group": {},
        "fpr_by_group": {},
    }
    if demographics is not None:
        g = _to_numpy(demographics)
        results["group_aucs"] = group_aucs(p, y, g)
        results["es_auc"] = equity_scaled_auc(p, y, g, alpha=alpha)
        results["dpd"] = demographic_parity_difference(p, g, threshold=threshold)
        deodds, tpr_g, fpr_g = equalized_odds_difference(p, y, g, threshold=threshold)
        results["deodds"] = deodds
        results["tpr_by_group"] = tpr_g
        results["fpr_by_group"] = fpr_g
    return results

def format_report(metrics: Dict[str, Any]) -> str:
    lines = [f"Overall AUC: {metrics['overall_auc']:.4f}"]
    if metrics.get("group_aucs"):
        lines.append("Group AUCs:")
        for grp, auc_val in metrics["group_aucs"].items():
            if auc_val is None or (isinstance(auc_val, float) and np.isnan(auc_val)):
                lines.append(f"  - Group {grp}: Undefined (only one class)")
            else:
                lines.append(f"  - Group {grp}: {auc_val:.4f}")
    if metrics.get("es_auc") is not None:
        lines.append(f"ES-AUC: {metrics['es_auc']:.4f}")
    if metrics.get("dpd") is not None:
        lines.append(f"DPD: {metrics['dpd']:.4f}")
    if metrics.get("deodds") is not None:
        lines.append(f"DEOdds: {metrics['deodds']:.4f}")
    return "\n".join(lines)


try:
    from tqdm import tqdm as _tqdm
except Exception:
    def _tqdm(x, **kwargs):
        return x


def evaluate_test_model(model, test_loader, device):
    model.eval()
    all_probs, all_labels, all_demographics = [], [], []
    has_demographics = False
    with torch.no_grad():
        for batch in _tqdm(test_loader, desc="Testing"):
            if len(batch) == 4:
                fundus_img, oct_tensor, demographics, labels = batch
                demographics = demographics.to(device)
                has_demographics = True
            else:
                fundus_img, oct_tensor, labels = batch
                demographics = None
            fundus_img, oct_tensor, labels = fundus_img.to(device), oct_tensor.to(device), labels.to(device)
            logits, _ = model(fundus_img, oct_tensor)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            if demographics is not None:
                all_demographics.append(demographics.cpu())
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_demographics = torch.cat(all_demographics).numpy() if has_demographics else None
    return all_probs, all_labels, all_demographics
