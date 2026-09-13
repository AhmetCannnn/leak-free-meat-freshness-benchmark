#!/usr/bin/env python3
"""Generate the review-ready nine-class resumable training notebook."""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path(__file__).resolve().parents[1] / "notebooks" / "meat_freshness_9class_multirun.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown(
        """# Nine-Class Meat Freshness Classification — Resumable Multi-Run Pipeline

This notebook trains MobileNetV3-Large, EfficientNet-B0, MobileNetV2, and ShuffleNetV2 on the leak-free nine-class benchmark. The authentic 36 h and 48 h Mutton classes remain separate; byte-identical 36/48 h Beef images remain represented as 36+ hr Beef.

The default configuration is a safe pilot: one model, one seed, and two epochs. For the complete four-model, five-seed experiment, run `python3 scripts/run_notebook_streaming.py --full`. The mode is selected externally so the notebook itself does not need to be edited.

Checkpointing is performed after every completed epoch. A resumed run restores the model, optimizer, scheduler, early-stopping state, random-number-generator states, completed epoch history, cumulative training/validation time, and observed peak memory. Interrupted time and incomplete epochs are not added to the reported duration.
"""
    ),
    markdown("## 1. Environment and Google Drive"),
    code(
        """import importlib.metadata
import importlib.util
import subprocess
import sys

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/gdrive", force_remount=False)

REQUIRED_TIMM_VERSION = "1.0.27"
required = {
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "seaborn": "seaborn",
    "matplotlib": "matplotlib",
    "scipy": "scipy",
}
missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
installed_timm = (
    importlib.metadata.version("timm") if importlib.util.find_spec("timm") is not None else None
)
if installed_timm != REQUIRED_TIMM_VERSION:
    missing.append(f"timm=={REQUIRED_TIMM_VERSION}")
if missing:
    print("Installing missing packages:", missing)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
else:
    print(f"All required packages are available; timm={installed_timm}.")
"""
    ),
    markdown("## 2. Imports, paths, and experiment configuration"),
    code(
        """from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import resource
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as tv_models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import binomtest, t
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

if timm.__version__ != REQUIRED_TIMM_VERSION:
    raise RuntimeError(
        f"Expected timm {REQUIRED_TIMM_VERSION}, found {timm.__version__}. "
        "Restart the runtime after the installation cell, then run all cells again."
    )

EXPERIMENT_VERSION = "nine-class-v1"
if IN_COLAB:
    # Keep persistent inputs/checkpoints in Drive, but train from Colab's local disk.
    PROJECT_DIR = Path("/content/gdrive/MyDrive/meat_freshness_9class_project")
    SOURCE_DATASET_DIR = PROJECT_DIR / "Clean_Dataset_9class_splits"
    DATASET_DIR = Path("/content/Clean_Dataset_9class_splits")
else:
    PROJECT_DIR = Path.cwd()
    SOURCE_DATASET_DIR = PROJECT_DIR / "Clean_Dataset_9class_splits"
    DATASET_DIR = SOURCE_DATASET_DIR

RUNS_DIR = PROJECT_DIR / "runs_9class"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

PILOT_MODE = os.environ.get("MEAT_FRESHNESS_FULL_RUN", "0") != "1"
ALLOW_CPU_TRAINING = False  # Change only if deliberately accepting a much slower CPU run.
PILOT_MODEL = "mobilenetv3_large_100"
PILOT_SEED = 42
PILOT_EPOCHS = 2

FULL_MODELS = [
    "mobilenetv3_large_100",
    "efficientnet_b0",
    "mobilenetv2_100",
    "shufflenet_v2_x1_0",
]
FULL_SEEDS = [42, 123, 2026, 3407, 9103]
FULL_EPOCHS = 35

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0  # Portable and deterministic on macOS and Colab.
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.05
PATIENCE = 6
FOCAL_GAMMA = 2.0
CONFIDENCE_LEVEL = 0.95

EXPECTED_CLASSES = [
    "0 hr Beef", "0 hr Mutton", "12 hr Beef", "12 hr Mutton",
    "24 hr Beef", "24 hr Mutton", "36 hr Mutton", "36+ hr Beef",
    "48 hr Mutton",
]
EXPECTED_SPLIT_SIZES = {"train": 1290, "valid": 277, "test": 282}

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

MODELS = [PILOT_MODEL] if PILOT_MODE else FULL_MODELS
SEEDS = [PILOT_SEED] if PILOT_MODE else FULL_SEEDS
MAX_EPOCHS = PILOT_EPOCHS if PILOT_MODE else FULL_EPOCHS
RUN_KIND = "pilot" if PILOT_MODE else "full"

print("Project:", PROJECT_DIR)
print("Persistent dataset source:", SOURCE_DATASET_DIR)
print("Dataset:", DATASET_DIR)
print("Runs:", RUNS_DIR)
print("Device:", DEVICE)
print("Mode:", RUN_KIND, "| Models:", MODELS, "| Seeds:", SEEDS, "| Epochs:", MAX_EPOCHS)
"""
    ),
    markdown("## 3. Dataset and protocol verification"),
    code(
        """IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset_against_manifest(root: Path) -> dict:
    manifest = root / "benchmark_manifest_md5.csv"
    report_path = root / "verification_report.json"
    if not root.is_dir() or not manifest.is_file() or not report_path.is_file():
        raise FileNotFoundError(f"Dataset, manifest, or verification report is missing under: {root}")

    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != sum(EXPECTED_SPLIT_SIZES.values()):
        raise ValueError(f"Manifest should contain 1849 rows; found {len(rows)}")

    manifest_paths = [row["relative_path"] for row in rows]
    if len(manifest_paths) != len(set(manifest_paths)):
        raise ValueError("Manifest contains repeated relative paths.")

    actual_paths = {
        path.relative_to(root).as_posix()
        for split in ("train", "valid", "test")
        for path in (root / split).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    expected_paths = set(manifest_paths)
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        raise ValueError(f"Dataset/manifest membership mismatch. Missing={missing[:5]}, extra={extra[:5]}")

    hash_locations = defaultdict(list)
    for row in rows:
        path = root / row["relative_path"]
        observed = md5_file(path)
        if observed != row["md5"]:
            raise ValueError(f"MD5 mismatch: {row['relative_path']}")
        if path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"File-size mismatch: {row['relative_path']}")
        hash_locations[observed].append(row["relative_path"])

    duplicate_groups = [paths for paths in hash_locations.values() if len(paths) > 1]
    if duplicate_groups:
        raise ValueError(f"Duplicate or cross-split MD5 group detected: {duplicate_groups[0]}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report["verification"]["dataset_clean"] or report["random_resplitting_performed"]:
        raise ValueError("Saved verification report does not describe a clean, fixed-split dataset.")
    return {
        "verified_files": len(rows),
        "manifest_md5": md5_file(manifest),
        "duplicate_md5_groups": 0,
    }


if IN_COLAB:
    print("Verifying the persistent Drive copy before local staging...")
    source_integrity = verify_dataset_against_manifest(SOURCE_DATASET_DIR)
    if not DATASET_DIR.exists():
        staging = Path("/content") / f".{DATASET_DIR.name}.staging-{os.getpid()}"
        try:
            shutil.copytree(SOURCE_DATASET_DIR, staging)
            verify_dataset_against_manifest(staging)
            os.replace(staging, DATASET_DIR)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        print("Dataset copied to Colab local storage:", DATASET_DIR)

DATASET_INTEGRITY = verify_dataset_against_manifest(DATASET_DIR)
if IN_COLAB and DATASET_INTEGRITY["manifest_md5"] != source_integrity["manifest_md5"]:
    raise ValueError("The Colab-local dataset does not match the verified Drive source manifest.")

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

eval_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

if not DATASET_DIR.is_dir():
    raise FileNotFoundError(
        f"Nine-class dataset not found: {DATASET_DIR}. "
        "In Colab, copy the dataset into PROJECT_DIR or update PROJECT_DIR explicitly."
    )

train_dataset = datasets.ImageFolder(DATASET_DIR / "train", transform=train_transforms)
valid_dataset = datasets.ImageFolder(DATASET_DIR / "valid", transform=eval_transforms)
test_dataset = datasets.ImageFolder(DATASET_DIR / "test", transform=eval_transforms)
class_names = train_dataset.classes

if class_names != EXPECTED_CLASSES:
    raise ValueError(f"Unexpected classes/order. Expected {EXPECTED_CLASSES}, found {class_names}")
actual_sizes = {
    "train": len(train_dataset), "valid": len(valid_dataset), "test": len(test_dataset)
}
if actual_sizes != EXPECTED_SPLIT_SIZES:
    raise ValueError(f"Unexpected split sizes. Expected {EXPECTED_SPLIT_SIZES}, found {actual_sizes}")

verification_path = DATASET_DIR / "verification_report.json"
verification = json.loads(verification_path.read_text(encoding="utf-8"))
if not verification["verification"]["dataset_clean"]:
    raise ValueError("The saved dataset verification report is not clean.")
if verification["random_resplitting_performed"]:
    raise ValueError("Unexpected re-splitting flag in verification report.")

train_class_counts = np.bincount(train_dataset.targets, minlength=len(class_names))
class_weights = torch.tensor(
    len(train_dataset) / (len(class_names) * train_class_counts),
    dtype=torch.float32,
    device=DEVICE,
)

print("Dataset verification passed.")
print("Classes:", class_names)
print("Split sizes:", actual_sizes)
print("Training counts:", dict(zip(class_names, train_class_counts.tolist())))
print("Class weights:", dict(zip(class_names, class_weights.cpu().tolist())))
print("Dataset integrity:", DATASET_INTEGRITY)
"""
    ),
    markdown("## 4. Reproducibility, timing, and memory helpers"),
    code(
        """def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # Loading a checkpoint with map_location=DEVICE also relocates RNG tensors.
    # The default CPU generator specifically requires a CPU ByteTensor.
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state["torch_cuda"]])


def synchronize_device() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif DEVICE.type == "mps":
        torch.mps.synchronize()


def process_peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    bytes_value = value if platform.system() == "Darwin" else value * 1024
    return float(bytes_value / 1_000_000)


def reset_accelerator_peak_memory() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif DEVICE.type == "mps":
        torch.mps.empty_cache()


def accelerator_memory_mb() -> float:
    if DEVICE.type == "cuda":
        return float(torch.cuda.max_memory_allocated() / 1_000_000)
    if DEVICE.type == "mps":
        # MPS has no CUDA-equivalent resettable max statistic; sample allocated memory.
        return float(torch.mps.current_allocated_memory() / 1_000_000)
    return 0.0


def atomic_torch_save(payload, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def atomic_json_save(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
    os.replace(temporary, destination)


def make_loader(dataset, *, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        generator=generator,
    )


def hardware_snapshot() -> dict:
    cuda_name = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None
    if DEVICE.type == "mps" and hasattr(torch.mps, "get_name"):
        mps_name = torch.mps.get_name(0)
    elif DEVICE.type == "mps":
        mps_name = f"Apple Silicon ({platform.machine()})"
    else:
        mps_name = None
    return {
        "timestamp_utc": utc_now(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "device_type": DEVICE.type,
        "cuda_device_name": cuda_name,
        "mps_device_name": mps_name,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "timm_version": timm.__version__,
    }


def hardware_compatibility_key(snapshot: dict) -> tuple:
    return (
        snapshot["device_type"], snapshot.get("cuda_device_name"),
        snapshot.get("mps_device_name"), snapshot["torch_version"],
        snapshot["torchvision_version"], snapshot["timm_version"],
    )


print(json.dumps(hardware_snapshot(), indent=2))
"""
    ),
    markdown("## 5. Model and loss definitions"),
    code(
        """class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def build_model(model_name: str, *, pretrained: bool) -> nn.Module:
    if model_name == "shufflenet_v2_x1_0":
        weights = tv_models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        model = tv_models.shufflenet_v2_x1_0(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
    else:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=len(class_names))
    # Full fine-tuning: no backbone layer is frozen.
    for parameter in model.parameters():
        parameter.requires_grad = True
    return model.to(DEVICE)


DISPLAY_NAMES = {
    "mobilenetv3_large_100": "MobileNetV3-Large",
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenetv2_100": "MobileNetV2",
    "shufflenet_v2_x1_0": "ShuffleNetV2",
}
"""
    ),
    markdown("## 6. Resumable training and final evaluation"),
    code(
        """def evaluate_test(model: nn.Module, run_dir: Path) -> dict:
    loader = make_loader(test_dataset, shuffle=False, seed=0)
    predictions, targets = [], []
    model.eval()
    synchronize_device()
    started = time.perf_counter()
    with torch.inference_mode():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            predictions.extend(outputs.argmax(1).cpu().tolist())
            targets.extend(labels.tolist())
    synchronize_device()
    test_time = time.perf_counter() - started

    report = classification_report(
        targets, predictions, target_names=class_names, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(targets, predictions)
    accuracy = accuracy_score(targets, predictions)

    prediction_path = run_dir / "test_predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "true_index", "true_class", "predicted_index", "predicted_class"])
        for (path, true_index), predicted_index in zip(test_dataset.samples, predictions):
            relative = Path(path).relative_to(DATASET_DIR).as_posix()
            writer.writerow([relative, true_index, class_names[true_index], predicted_index, class_names[predicted_index]])

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"{run_dir.parent.name} | {run_dir.name} | Accuracy: {accuracy * 100:.2f}%")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(run_dir / "confusion_matrix.png", dpi=300)
    plt.close(fig)

    return {
        "test_accuracy": accuracy,
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {name: report[name] for name in class_names},
        "confusion_matrix": matrix.tolist(),
        "test_evaluation_seconds": test_time,
    }


PROTECTED_CONFIG_KEYS = [
    "experiment_version", "run_kind", "model", "seed", "classes", "num_classes",
    "split_sizes", "dataset_manifest_md5", "image_size", "batch_size", "num_workers",
    "learning_rate", "weight_decay", "max_epochs", "patience", "focal_gamma",
    "augmentation", "class_weights", "selection_metric", "full_network_fine_tuning",
]


def assert_config_compatible(saved: dict, expected: dict, context: str) -> None:
    missing = [key for key in PROTECTED_CONFIG_KEYS if key not in saved]
    mismatches = [
        key for key in PROTECTED_CONFIG_KEYS
        if key in saved and saved[key] != expected[key]
    ]
    if missing or mismatches:
        raise ValueError(
            f"{context} is incompatible with the current experiment. "
            f"Missing keys={missing}; mismatched keys={mismatches}. "
            "Move the old run directory aside; do not mix results from different protocols."
        )


def run_training(model_name: str, seed: int) -> dict:
    if DEVICE.type == "cpu" and not ALLOW_CPU_TRAINING:
        raise RuntimeError(
            "No CUDA/MPS accelerator is active. Training was stopped to avoid an accidental "
            "long CPU run. Enable a Colab GPU or deliberately set ALLOW_CPU_TRAINING = True."
        )

    run_dir = RUNS_DIR / RUN_KIND / model_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint_last.pth"
    best_model_path = run_dir / "best_model.pth"
    completed_path = run_dir / "test_results.json"
    config_path = run_dir / "run_config.json"
    current_hardware = hardware_snapshot()
    config = {
        "experiment_version": EXPERIMENT_VERSION,
        "run_kind": RUN_KIND,
        "model": model_name,
        "display_name": DISPLAY_NAMES[model_name],
        "seed": seed,
        "classes": class_names,
        "num_classes": len(class_names),
        "split_sizes": EXPECTED_SPLIT_SIZES,
        "dataset_manifest_md5": DATASET_INTEGRITY["manifest_md5"],
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "focal_gamma": FOCAL_GAMMA,
        "class_weights": class_weights.detach().cpu().tolist(),
        "augmentation": {
            "horizontal_flip_probability": 0.5,
            "vertical_flip_probability": 0.5,
            "rotation_degrees": 15,
            "brightness": 0.15,
            "contrast": 0.15,
            "saturation": 0.15,
            "hue": 0.02,
            "gaussian_blur_probability": 0.2,
        },
        "hue_selection_note": (
            "The hue bound of 0.02 is a conservative domain-informed choice intended to avoid "
            "distorting the myoglobin-related color signal; no hue ablation is claimed."
        ),
        "selection_metric": "minimum validation loss",
        "full_network_fine_tuning": True,
        "memory_measurement_note": (
            "Accelerator memory is sampled during training/validation. Process peak RSS is a "
            "kernel-lifetime diagnostic and must not be compared across sequential runs."
        ),
        "hardware": current_hardware,
    }

    if completed_path.is_file():
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        assert_config_compatible(completed, config, f"Completed result {completed_path}")
        print(f"SKIP verified completed run: {model_name}, seed={seed}")
        return completed

    seed_everything(seed)
    model = build_model(model_name, pretrained=not checkpoint_path.is_file())
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights)

    start_epoch = 0
    best_val_loss = math.inf
    best_val_accuracy = 0.0
    best_model_state_dict = None
    epochs_no_improve = 0
    cumulative_training_validation_seconds = 0.0
    observed_peak_accelerator_memory_mb = 0.0
    observed_peak_process_rss_mb = process_peak_rss_mb()
    history = []

    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        saved_config = checkpoint["config"]
        assert_config_compatible(saved_config, config, f"Checkpoint {checkpoint_path}")
        if hardware_compatibility_key(saved_config["hardware"]) != hardware_compatibility_key(current_hardware):
            raise RuntimeError(
                "The accelerator or software environment changed since this run was checkpointed. "
                "Resume on the original environment so that time and memory measurements remain comparable."
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["completed_epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        best_val_accuracy = checkpoint["best_val_accuracy"]
        best_model_state_dict = checkpoint["best_model_state_dict"]
        epochs_no_improve = checkpoint["epochs_no_improve"]
        cumulative_training_validation_seconds = checkpoint["cumulative_training_validation_seconds"]
        observed_peak_accelerator_memory_mb = checkpoint["observed_peak_accelerator_memory_mb"]
        observed_peak_process_rss_mb = checkpoint["observed_peak_process_rss_mb"]
        history = checkpoint["history"]
        restore_rng_state(checkpoint.get("rng_state"))
        print(f"RESUME {model_name}, seed={seed}, next epoch={start_epoch + 1}")
        if epochs_no_improve >= PATIENCE:
            print("Early stopping had already been reached; proceeding directly to test evaluation.")
            start_epoch = MAX_EPOCHS
    else:
        atomic_json_save(config, config_path)
        print(f"START {model_name}, seed={seed}")

    reset_accelerator_peak_memory()
    for epoch in range(start_epoch, MAX_EPOCHS):
        # A deterministic epoch seed makes resumed and uninterrupted runs use the same order/augmentations.
        epoch_seed = seed * 100_000 + epoch
        seed_everything(epoch_seed)
        train_loader = make_loader(train_dataset, shuffle=True, seed=epoch_seed)
        valid_loader = make_loader(valid_dataset, shuffle=False, seed=seed)

        synchronize_device()
        epoch_started = time.perf_counter()
        learning_rate_used = optimizer.param_groups[0]["lr"]
        model.train()
        train_loss_sum = 0.0
        peak_accelerator_this_epoch = accelerator_memory_mb()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * inputs.size(0)
            peak_accelerator_this_epoch = max(peak_accelerator_this_epoch, accelerator_memory_mb())

        model.eval()
        valid_loss_sum = 0.0
        valid_predictions, valid_targets = [], []
        with torch.inference_mode():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                valid_loss_sum += criterion(outputs, labels).item() * inputs.size(0)
                valid_predictions.extend(outputs.argmax(1).cpu().tolist())
                valid_targets.extend(labels.tolist())
                peak_accelerator_this_epoch = max(peak_accelerator_this_epoch, accelerator_memory_mb())

        scheduler.step()
        synchronize_device()
        epoch_seconds = time.perf_counter() - epoch_started
        cumulative_training_validation_seconds += epoch_seconds

        train_loss = train_loss_sum / len(train_dataset)
        valid_loss = valid_loss_sum / len(valid_dataset)
        valid_accuracy = accuracy_score(valid_targets, valid_predictions)
        improved = valid_loss < best_val_loss
        if improved:
            best_val_loss = valid_loss
            best_val_accuracy = valid_accuracy
            epochs_no_improve = 0
            best_model_state_dict = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            atomic_torch_save(best_model_state_dict, best_model_path)
        else:
            epochs_no_improve += 1

        observed_peak_accelerator_memory_mb = max(
            observed_peak_accelerator_memory_mb, peak_accelerator_this_epoch
        )
        observed_peak_process_rss_mb = max(observed_peak_process_rss_mb, process_peak_rss_mb())
        history.append({
            "epoch": epoch + 1,
            "epoch_seed": epoch_seed,
            "train_loss": train_loss,
            "validation_loss": valid_loss,
            "validation_accuracy": valid_accuracy,
            "epoch_training_validation_seconds": epoch_seconds,
            "cumulative_training_validation_seconds": cumulative_training_validation_seconds,
            "learning_rate": learning_rate_used,
            "improved_validation_loss": improved,
            "epochs_no_improve": epochs_no_improve,
            "accelerator_memory_mb_observed": peak_accelerator_this_epoch,
            "process_lifetime_peak_rss_mb_diagnostic": observed_peak_process_rss_mb,
        })

        checkpoint = {
            "format_version": 1,
            "saved_at_utc": utc_now(),
            "config": config,
            "completed_epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_val_accuracy": best_val_accuracy,
            "best_model_state_dict": best_model_state_dict,
            "epochs_no_improve": epochs_no_improve,
            "cumulative_training_validation_seconds": cumulative_training_validation_seconds,
            "observed_peak_accelerator_memory_mb": observed_peak_accelerator_memory_mb,
            "observed_peak_process_rss_mb": observed_peak_process_rss_mb,
            "history": history,
            "rng_state": capture_rng_state(),
        }
        atomic_torch_save(checkpoint, checkpoint_path)
        pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)

        print(
            f"{DISPLAY_NAMES[model_name]} | seed {seed} | epoch {epoch + 1}/{MAX_EPOCHS} | "
            f"train loss {train_loss:.4f} | val loss {valid_loss:.4f} | "
            f"val acc {valid_accuracy:.4f} | {epoch_seconds:.1f}s | checkpoint saved"
        )
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    if best_model_state_dict is None:
        raise RuntimeError("No best-model state is available after training/resume.")
    model.load_state_dict(best_model_state_dict)
    test_metrics = evaluate_test(model, run_dir)
    result = {
        **config,
        "completed_at_utc": utc_now(),
        "completed_epochs": len(history),
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "cumulative_training_validation_seconds": cumulative_training_validation_seconds,
        "mean_completed_epoch_seconds": float(np.mean([row["epoch_training_validation_seconds"] for row in history])),
        "observed_peak_accelerator_memory_mb": observed_peak_accelerator_memory_mb,
        "process_lifetime_peak_rss_mb_diagnostic": observed_peak_process_rss_mb,
        **test_metrics,
    }
    atomic_json_save(result, completed_path)
    print(f"COMPLETE {model_name}, seed={seed}, test accuracy={result['test_accuracy']:.4f}")
    return result
"""
    ),
    markdown(
        """## 7. Start or resume configured runs

Running this cell starts training. With the default settings, only the two-epoch pilot for MobileNetV3-Large and seed 42 is executed. Re-running the cell resumes an incomplete run or skips a completed run.
"""
    ),
    code(
        """all_results = []
for model_name in MODELS:
    for seed in SEEDS:
        all_results.append(run_training(model_name, seed))

print(f"Configured runs completed or recovered: {len(all_results)}")
"""
    ),
    markdown("## 8. Paper-ready aggregation, uncertainty, and significance analysis"),
    code(
        """def mean_sd_ci(values, confidence=CONFIDENCE_LEVEL):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    if len(values) > 1:
        margin = float(t.ppf((1 + confidence) / 2, df=len(values) - 1) * sd / np.sqrt(len(values)))
        return mean, sd, mean - margin, mean + margin
    return mean, sd, float("nan"), float("nan")


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


result_files = sorted((RUNS_DIR / "full").glob("*/seed_*/test_results.json")) if (RUNS_DIR / "full").exists() else []
records = [json.loads(path.read_text(encoding="utf-8")) for path in result_files]

if not records:
    print("No completed full runs yet. Pilot results are intentionally excluded from the paper summary.")
else:
    expected_runs = {(model, seed) for model in FULL_MODELS for seed in FULL_SEEDS}
    record_keys = [(row.get("model"), row.get("seed")) for row in records]
    completed_runs = set(record_keys)
    duplicate_keys = sorted({key for key in record_keys if record_keys.count(key) > 1})
    missing_runs = sorted(expected_runs - completed_runs)
    unexpected_runs = sorted(completed_runs - expected_runs)
    if duplicate_keys or missing_runs or unexpected_runs or len(records) != len(expected_runs):
        raise RuntimeError(
            "PAPER-READY OUTPUT BLOCKED. "
            f"Expected exactly 20 runs; found {len(records)}. "
            f"Missing={missing_runs}; unexpected={unexpected_runs}; duplicates={duplicate_keys}."
        )

    expected_common = {
        "experiment_version": EXPERIMENT_VERSION,
        "run_kind": "full",
        "classes": class_names,
        "num_classes": len(class_names),
        "split_sizes": EXPECTED_SPLIT_SIZES,
        "dataset_manifest_md5": DATASET_INTEGRITY["manifest_md5"],
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": FULL_EPOCHS,
        "patience": PATIENCE,
        "focal_gamma": FOCAL_GAMMA,
        "class_weights": class_weights.detach().cpu().tolist(),
        "augmentation": {
            "horizontal_flip_probability": 0.5,
            "vertical_flip_probability": 0.5,
            "rotation_degrees": 15,
            "brightness": 0.15,
            "contrast": 0.15,
            "saturation": 0.15,
            "hue": 0.02,
            "gaussian_blur_probability": 0.2,
        },
        "selection_metric": "minimum validation loss",
        "full_network_fine_tuning": True,
    }
    for record in records:
        mismatches = [key for key, value in expected_common.items() if record.get(key) != value]
        if mismatches:
            raise ValueError(
                f"Saved result for {record.get('model')}, seed={record.get('seed')} "
                f"does not match the current protocol: {mismatches}"
            )

    environment_keys = {hardware_compatibility_key(row["hardware"]) for row in records}
    if len(environment_keys) != 1:
        raise RuntimeError(
            "PAPER-READY OUTPUT BLOCKED: full runs used different accelerator/software environments. "
            "Time and memory values would not be directly comparable."
        )

    print("PAPER-READY COMPLETENESS CHECK PASSED: all 4 models × 5 seeds are present and compatible.")
    paper_ready_dir = RUNS_DIR / "paper_ready"
    paper_ready_dir.mkdir(parents=True, exist_ok=True)

    raw_columns = [
        "display_name", "seed", "completed_epochs", "test_accuracy", "macro_precision",
        "macro_recall", "macro_f1", "cumulative_training_validation_seconds",
        "mean_completed_epoch_seconds", "observed_peak_accelerator_memory_mb",
        "process_lifetime_peak_rss_mb_diagnostic",
    ]
    raw = pd.DataFrame(records)[raw_columns].sort_values(["display_name", "seed"])
    raw.to_csv(paper_ready_dir / "full_run_results.csv", index=False)
    display(raw)

    metric_columns = [
        "test_accuracy", "macro_precision", "macro_recall", "macro_f1",
        "cumulative_training_validation_seconds", "mean_completed_epoch_seconds",
        "observed_peak_accelerator_memory_mb",
    ]
    summary = raw.groupby("display_name")[metric_columns].agg(["mean", "std", "count"])
    summary.to_csv(paper_ready_dir / "full_run_summary_mean_sd.csv")
    display(summary)

    # Main manuscript table: all metrics come from the same five saved test predictions.
    paper_rows = []
    for model_name, group in raw.groupby("display_name", sort=False):
        row = {"Model": model_name}
        for metric, label in [
            ("test_accuracy", "Accuracy"), ("macro_precision", "Macro precision"),
            ("macro_recall", "Macro recall"), ("macro_f1", "Macro F1"),
        ]:
            mean, sd, low, high = mean_sd_ci(group[metric].to_numpy() * 100)
            row[f"{label} mean (%)"] = mean
            row[f"{label} SD (%)"] = sd
            row[f"{label} 95% CI low (%)"] = low
            row[f"{label} 95% CI high (%)"] = high
            row[f"{label} manuscript"] = f"{mean:.2f} ± {sd:.2f}"
        paper_rows.append(row)
    paper_table = pd.DataFrame(paper_rows)
    paper_table.to_csv(paper_ready_dir / "paper_table_overall_metrics.csv", index=False)
    display(paper_table)

    # Training efficiency requested by Reviewer 1 Comment 2.
    efficiency_rows = []
    for model_name, group in raw.groupby("display_name", sort=False):
        row = {"Model": model_name}
        for metric, label in [
            ("completed_epochs", "Completed epochs"),
            ("cumulative_training_validation_seconds", "Training + validation time (s)"),
            ("mean_completed_epoch_seconds", "Mean epoch time (s)"),
            ("observed_peak_accelerator_memory_mb", "Peak accelerator memory (MB)"),
        ]:
            mean, sd, low, high = mean_sd_ci(group[metric].to_numpy())
            row[f"{label} mean"] = mean
            row[f"{label} SD"] = sd
            row[f"{label} 95% CI low"] = low
            row[f"{label} 95% CI high"] = high
            row[f"{label} manuscript"] = f"{mean:.2f} ± {sd:.2f}"
        efficiency_rows.append(row)
    efficiency_table = pd.DataFrame(efficiency_rows)
    efficiency_table.to_csv(paper_ready_dir / "paper_table_training_efficiency.csv", index=False)
    display(efficiency_table)

    # Ranking stability across the same five seeds, for Reviewer 3 Major 2 / Question 4.
    ranking_rows = []
    for metric in ["test_accuracy", "macro_f1"]:
        pivot = raw.pivot(index="seed", columns="display_name", values=metric)
        ranks = pivot.rank(axis=1, ascending=False, method="min")
        for model_name in pivot.columns:
            ranking_rows.append({
                "Metric": metric,
                "Model": model_name,
                "Mean rank": float(ranks[model_name].mean()),
                "Rank SD": float(ranks[model_name].std(ddof=1)),
                "First-place seeds": int((ranks[model_name] == 1).sum()),
                "Seeds evaluated": int(len(ranks)),
            })
        seed_table = pivot.copy()
        seed_table.columns = [f"{column} value" for column in seed_table.columns]
        rank_columns = ranks.copy()
        rank_columns.columns = [f"{column} rank" for column in rank_columns.columns]
        seed_table.join(rank_columns).to_csv(paper_ready_dir / f"per_seed_ranking_{metric}.csv")
    ranking_summary = pd.DataFrame(ranking_rows)
    ranking_summary.to_csv(paper_ready_dir / "model_ranking_stability.csv", index=False)
    display(ranking_summary)

    # Reviewer 3 Minor 8: per-class precision, recall, and F1 as mean ± SD across seeds.
    per_class_rows = []
    for record in records:
        for class_name in class_names:
            metrics = record["per_class"][class_name]
            per_class_rows.append({
                "Model": record["display_name"], "Seed": record["seed"], "Class": class_name,
                "Precision": metrics["precision"], "Recall": metrics["recall"],
                "F1": metrics["f1-score"], "Support": metrics["support"],
            })
    per_class_raw = pd.DataFrame(per_class_rows)
    per_class_raw.to_csv(paper_ready_dir / "per_class_metrics_all_runs.csv", index=False)
    per_class_paper_rows = []
    for (model_name, class_name), group in per_class_raw.groupby(["Model", "Class"], sort=False):
        supports = group["Support"].unique()
        if len(supports) != 1:
            raise ValueError(f"Inconsistent test support for {model_name}, {class_name}")
        row = {"Model": model_name, "Class": class_name, "Support": int(supports[0])}
        for metric in ["Precision", "Recall", "F1"]:
            mean, sd, _, _ = mean_sd_ci(group[metric].to_numpy() * 100)
            row[f"{metric} mean (%)"] = mean
            row[f"{metric} SD (%)"] = sd
            row[f"{metric} manuscript"] = f"{mean:.2f} ± {sd:.2f}"
        per_class_paper_rows.append(row)
    per_class_summary = pd.DataFrame(per_class_paper_rows)
    per_class_summary.to_csv(paper_ready_dir / "paper_table_per_class_mean_sd.csv", index=False)
    display(per_class_summary)

    # Reviewer 2: exact paired McNemar tests on shared test images, separately for each seed.
    mcnemar_rows = []
    for seed in FULL_SEEDS:
        seed_records = {row["model"]: row for row in records if row["seed"] == seed}
        seed_tests = []
        for model_a, model_b in combinations(FULL_MODELS, 2):
            path_a = RUNS_DIR / "full" / model_a / f"seed_{seed}" / "test_predictions.csv"
            path_b = RUNS_DIR / "full" / model_b / f"seed_{seed}" / "test_predictions.csv"
            a = pd.read_csv(path_a).sort_values("relative_path").reset_index(drop=True)
            b = pd.read_csv(path_b).sort_values("relative_path").reset_index(drop=True)
            if not a[["relative_path", "true_index"]].equals(b[["relative_path", "true_index"]]):
                raise ValueError(f"Prediction alignment failed for seed {seed}: {model_a} vs {model_b}")
            correct_a = a["predicted_index"].to_numpy() == a["true_index"].to_numpy()
            correct_b = b["predicted_index"].to_numpy() == b["true_index"].to_numpy()
            b_count = int(np.sum(correct_a & ~correct_b))
            c_count = int(np.sum(~correct_a & correct_b))
            discordant = b_count + c_count
            p_value = 1.0 if discordant == 0 else float(binomtest(b_count, discordant, 0.5).pvalue)
            seed_tests.append({
                "seed": seed, "model_a": DISPLAY_NAMES[model_a], "model_b": DISPLAY_NAMES[model_b],
                "a_correct_b_wrong": b_count, "a_wrong_b_correct": c_count,
                "discordant_pairs": discordant, "exact_p": p_value,
            })
        adjusted = holm_adjust([row["exact_p"] for row in seed_tests])
        for row, adjusted_p in zip(seed_tests, adjusted):
            row["holm_adjusted_p_within_seed"] = float(adjusted_p)
            row["significant_at_0.05"] = bool(adjusted_p < 0.05)
        mcnemar_rows.extend(seed_tests)
    mcnemar_table = pd.DataFrame(mcnemar_rows)
    mcnemar_table.to_csv(paper_ready_dir / "pairwise_mcnemar_exact_holm.csv", index=False)
    display(mcnemar_table)

    mcnemar_summary = (
        mcnemar_table.groupby(["model_a", "model_b"], as_index=False)
        .agg(
            seeds_tested=("seed", "count"),
            significant_seeds=("significant_at_0.05", "sum"),
            median_exact_p=("exact_p", "median"),
            median_holm_p=("holm_adjusted_p_within_seed", "median"),
        )
    )
    mcnemar_summary.to_csv(paper_ready_dir / "mcnemar_summary_across_seeds.csv", index=False)
    display(mcnemar_summary)

    # Figure 16 replacement: mean row-normalized confusion matrices across the same five runs.
    fig, axes = plt.subplots(2, 2, figsize=(18, 16), constrained_layout=True)
    for ax, model in zip(axes.flat, FULL_MODELS):
        model_records = [row for row in records if row["model"] == model]
        matrices = np.asarray([row["confusion_matrix"] for row in model_records], dtype=float)
        normalized = matrices / matrices.sum(axis=2, keepdims=True)
        mean_percent = normalized.mean(axis=0) * 100
        sns.heatmap(mean_percent, annot=True, fmt=".1f", vmin=0, vmax=100, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=False)
        mean_acc, sd_acc, _, _ = mean_sd_ci([row["test_accuracy"] * 100 for row in model_records])
        ax.set_title(f"{DISPLAY_NAMES[model]} | Accuracy {mean_acc:.2f} ± {sd_acc:.2f}%")
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.tick_params(axis="x", rotation=40)
        ax.tick_params(axis="y", rotation=0)
    figure_path = paper_ready_dir / "figure16_mean_normalized_confusion_matrices.png"
    fig.savefig(figure_path, dpi=600, bbox_inches="tight")
    plt.show()

    paper_ready_manifest = {
        "created_at_utc": utc_now(),
        "experiment_version": EXPERIMENT_VERSION,
        "dataset_manifest_md5": DATASET_INTEGRITY["manifest_md5"],
        "completed_runs": len(records),
        "models": FULL_MODELS,
        "seeds": FULL_SEEDS,
        "hardware_compatibility_key": list(next(iter(environment_keys))),
        "figure": figure_path.name,
    }
    atomic_json_save(paper_ready_manifest, paper_ready_dir / "PAPER_READY.json")
    print("All validated paper-ready outputs saved under:", paper_ready_dir)
"""
    ),
    markdown(
        """## 9. Operational notes

- Do not change dataset membership, class order, hyperparameters, or seed list after full runs begin.
- Colab reads images from a verified local copy; checkpoints and results remain persistent in Drive.
- Resume is blocked if the accelerator or core software versions differ from the saved run.
- An epoch interrupted before checkpoint completion is rerun from its beginning and is not counted in cumulative time.
- Pilot outputs are stored under `runs_9class/pilot` and are never included in the full-run paper summary.
- The final paper should report the exact hardware/software environment and mean ± standard deviation across completed full runs.
- Paper-ready tables and figures are blocked until exactly 20 compatible full runs are present.
- Report exact paired McNemar tests with Holm correction as generated here; do not describe the experiment as five-fold cross-validation.
- The hue range is a conservative domain-informed setting. Unless a separate ablation is run, state explicitly that it was not selected through an empirical ablation.
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Generated: {OUTPUT}")
