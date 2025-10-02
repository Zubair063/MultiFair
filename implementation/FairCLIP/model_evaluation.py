
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
from fairlearn.metrics import MetricFrame, true_positive_rate, false_positive_rate

class FairnessEvaluator:
    def __init__(self, model, model_path, device):
        self.model = model.to(device)
        self.device = device
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def evaluate(self, test_loader):
        self.model.eval()
        all_probs = []
        all_labels = []
        all_demographics = []
        has_demographics = False

        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 4:
                    fundus_img, text_data, demographics, labels = batch
                    text_data = {k: v.to(self.device) for k, v in text_data.items()}
                    has_demographics = True
                    logits, _ = self.model(fundus_img.to(self.device), text_data)
                elif len(batch) == 3:
                    fundus_img, demographics, labels = batch
                    has_demographics = True
                    logits = self.model(fundus_img.to(self.device))
                else:
                    fundus_img, labels = batch
                    logits = self.model(fundus_img.to(self.device))
                    demographics = None

                probs = torch.softmax(logits, dim=1)[:, 1]
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())
                if demographics is not None:
                    all_demographics.append(demographics.cpu())

        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_demographics = torch.cat(all_demographics).view(-1).numpy() if has_demographics else None

        return all_probs, all_labels, all_demographics

    def demographic_group_aucs(self, probs, labels, demographics):
        group_aucs = {}
        for g in np.unique(demographics):
            m = (demographics == g)
            y_g = labels[m]
            p_g = probs[m]
            if len(np.unique(y_g)) < 2:
                group_aucs[g] = np.nan
            else:
                group_aucs[g] = roc_auc_score(y_g, p_g)
        return group_aucs

    def equity_scaled_auc(self, probs, labels, demographics, alpha=1.0):
        fpr, tpr, _ = roc_curve(labels, probs)
        overall_auc = auc(fpr, tpr)
        identity_wise_perf = []

        for group in np.unique(demographics):
            group_mask = demographics == group
            group_probs = probs[group_mask]
            group_labels = labels[group_mask]
            if len(np.unique(group_labels)) < 2:
                continue
            fpr_g, tpr_g, _ = roc_curve(group_labels, group_probs)
            group_auc = auc(fpr_g, tpr_g)
            identity_wise_perf.append(group_auc)

        total_deviation = sum(np.abs(g_auc - overall_auc) for g_auc in identity_wise_perf)
        es_auc = overall_auc / (alpha * total_deviation + 1)
        return es_auc

    def demographic_parity_difference(self, probs, demographics, threshold=0.5):
        y_pred = (probs >= threshold).astype(int)
        groups = np.unique(demographics)
        selection_rates = {g: np.mean(y_pred[demographics == g]) for g in groups}
        dpd = max(abs(selection_rates[g1] - selection_rates[g2])
                  for i, g1 in enumerate(groups)
                  for g2 in groups[i + 1:])
        return dpd

    def compute_deodds_fairlearn(self, probs, labels, demographics, threshold=0.5):
        y_pred = (probs >= threshold).astype(int)
        frame = MetricFrame(
            metrics={"TPR": true_positive_rate, "FPR": false_positive_rate},
            y_true=labels,
            y_pred=y_pred,
            sensitive_features=demographics,
        )
        tpr_diff = frame.difference(method="between_groups")["TPR"]
        fpr_diff = frame.difference(method="between_groups")["FPR"]
        deodds = max(tpr_diff, fpr_diff)
        return deodds

    def compute_all_metrics(self, test_loader, alpha=1.0, threshold=0.5):
        probs, labels, demographics = self.evaluate(test_loader)
        auc_score = roc_auc_score(labels, probs)
        es_auc = self.equity_scaled_auc(probs, labels, demographics, alpha)
        dpd = self.demographic_parity_difference(probs, demographics, threshold)
        deodds = self.compute_deodds_fairlearn(probs, labels, demographics, threshold)
        group_aucs = self.demographic_group_aucs(probs, labels, demographics)
        return {"AUC": auc_score, "ES-AUC": es_auc, "DPD": dpd, "DEOdds": deodds, "GroupAUCs": group_aucs}
