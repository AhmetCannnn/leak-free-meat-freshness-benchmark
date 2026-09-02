#!/usr/bin/env python3
"""Evaluate released model weights on the canonical test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import timm
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

MODEL_SPECS = (
    ("mobilenetv3_large_100", "MobileNetV3-Large"),
    ("efficientnet_b0", "EfficientNet-B0"),
    ("mobilenetv2_100", "MobileNetV2"),
    ("shufflenet_v2_x1_0", "ShuffleNetV2"),
)


def create_model(name: str, classes: int) -> nn.Module:
    if name == "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0(weights=None)
        model.fc = nn.Linear(model.fc.in_features, classes)
        return model
    return timm.create_model(name, pretrained=False, num_classes=classes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(args.dataset_root / "test", transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    summaries, per_class, matrices = [], [], {}

    for model_name, display_name in MODEL_SPECS:
        checkpoint = args.checkpoint_root / f"{model_name}_best_final.pth"
        if not checkpoint.is_file():
            print(f"Skipping missing checkpoint: {checkpoint}")
            continue
        model = create_model(model_name, len(dataset.classes))
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.to(device).eval()
        targets, predictions = [], []
        with torch.inference_mode():
            for images, labels in loader:
                predictions.extend(model(images.to(device)).argmax(1).cpu().tolist())
                targets.extend(labels.tolist())
        report = classification_report(targets, predictions, target_names=dataset.classes, output_dict=True, zero_division=0)
        summaries.append({
            "model": display_name,
            "accuracy": accuracy_score(targets, predictions),
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_precision": report["weighted avg"]["precision"],
            "weighted_recall": report["weighted avg"]["recall"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "test_samples": len(dataset),
        })
        for class_name in dataset.classes:
            per_class.append({"model": display_name, "class": class_name, **report[class_name]})
        matrices[display_name] = confusion_matrix(targets, predictions).tolist()

    args.output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("model_performance.csv", summaries), ("per_class_metrics.csv", per_class)):
        if rows:
            with (args.output / filename).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
    (args.output / "confusion_matrices.json").write_text(json.dumps(matrices, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

