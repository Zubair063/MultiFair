#!/usr/bin/env python3
import torch
from model_evaluation import FairnessEvaluator
from fairclip_dataloader import build_bimodal_dataloaders
from models_module import GlaucomaBiModel

DATA_ROOT = "/FairCLIP"
CSV_PATH  = "/FairCLIP/data_summary.csv"
MODEL_PATH = "multifair_fairclip_model.pth"

def main():
    train_loader, val_loader, test_loader = build_bimodal_dataloaders(
        data_root=DATA_ROOT,
        csv_path=CSV_PATH,
        batch_size=32,
        num_workers=4,
        demographic_key="gender"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GlaucomaBiModel(
        output_dim=2,
        proj_dim=256,
        fusion_heads=8,
        fusion_layers=3,
        dropout=0.1,
        vit_variant='google/vit-base-patch16-224-in21k',
        pretrained=True
    ).to(device)

    evaluator = FairnessEvaluator(model, MODEL_PATH, device)

    probs, labels, demographics = evaluator.evaluate(test_loader)
    metrics = evaluator.compute_all_metrics(test_loader, alpha=1.0, threshold=0.5)
    print("AUC:", metrics["AUC"])
    print("ES-AUC:", metrics["ES-AUC"])
    print("DPD:", metrics["DPD"])
    print("DEOdds:", metrics["DEOdds"])
    print("GroupAUCs:", metrics["GroupAUCs"])

if __name__ == "__main__":
    main()
