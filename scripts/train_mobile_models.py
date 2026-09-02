#!/usr/bin/env python3
"""Train the four mobile architectures using the manuscript protocol."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

MODEL_NAMES = ("mobilenetv3_large_100", "efficientnet_b0", "mobilenetv2_100", "shufflenet_v2_x1_0")


class FocalLoss(nn.Module):
    def __init__(self, gamma: float, weight: torch.Tensor):
        super().__init__(); self.gamma = gamma; self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        return (((1 - torch.exp(-ce)) ** self.gamma) * ce).mean()


def make_model(name: str, classes: int) -> nn.Module:
    if name == "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, classes)
        return model
    return timm.create_model(name, pretrained=True, num_classes=classes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5), transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_set = datasets.ImageFolder(args.dataset_root / "train", transform=train_tf)
    valid_set = datasets.ImageFolder(args.dataset_root / "valid", transform=eval_tf)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=0, generator=generator)
    valid_loader = DataLoader(valid_set, args.batch_size, shuffle=False, num_workers=0)
    counts = np.bincount([label for _, label in train_set.samples], minlength=len(train_set.classes))
    weights = torch.tensor(len(train_set) / (len(train_set.classes) * counts), dtype=torch.float32, device=device)
    args.output.mkdir(parents=True, exist_ok=True)

    run_summary = {}
    for name in MODEL_NAMES:
        model = make_model(name, len(train_set.classes)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = FocalLoss(2.0, weights)
        best_loss, stale = float("inf"), 0
        history = []
        for epoch in range(args.epochs):
            model.train(); train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(); loss = criterion(model(images), labels)
                loss.backward(); optimizer.step(); train_loss += loss.item() * len(labels)
            scheduler.step(); model.eval(); valid_loss = 0.0; targets = []; predictions = []
            with torch.inference_mode():
                for images, labels in valid_loader:
                    images, labels = images.to(device), labels.to(device); logits = model(images)
                    valid_loss += criterion(logits, labels).item() * len(labels)
                    targets.extend(labels.cpu().tolist()); predictions.extend(logits.argmax(1).cpu().tolist())
            valid_loss /= len(valid_set)
            row = {"epoch": epoch + 1, "train_loss": train_loss / len(train_set), "valid_loss": valid_loss, "valid_accuracy": accuracy_score(targets, predictions)}
            history.append(row); print(name, row)
            if valid_loss < best_loss:
                best_loss, stale = valid_loss, 0
                torch.save(model.state_dict(), args.output / f"{name}_best_final.pth")
            else:
                stale += 1
                if stale >= args.patience: break
        run_summary[name] = history
    (args.output / "training_history.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

