#!/usr/bin/env python3
import torch
from multifair_models import GlaucomaViTModel, ClassifierGuided
from fair_metrics import evaluate_test_model, evaluate_all, format_report
from oct_fundus_dataloader import OCTTransform, GlaucomaOCTFundusDataset, build_oct_fundus_dataloaders


def main():
    DATA_ROOT = "/medailab/medailab/shilab/FairVision/Glaucoma"

    train_loader, val_loader, test_loader = build_oct_fundus_dataloaders(
        data_root=DATA_ROOT,
        batch_size=32,
        num_workers=4,
        demographic_key="male",
        transform_oct=OCTTransform(target_size=(224, 224), normalize=True)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GlaucomaViTModel(
        output_dim=2,
        proj_dim=128,
        fusion_heads=4,
        fusion_layers=2,
        dropout=0.1,
        vit_variant='google/vit-base-patch16-224-in21k',
        pretrained=True
    ).to(device)

    model_path = "fairness_aware_cggm_vit_model_auc.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}")

    probs, labels, demographics = evaluate_test_model(model, test_loader, device)

    if demographics is not None and demographics.ndim > 1:
        demographics = demographics.squeeze(1)

    metrics = evaluate_all(probs, labels, demographics, threshold=0.5, alpha=1.0)
    print(format_report(metrics))


if __name__ == "__main__":
    main()
