"""
柴犬 vs 柴犬以外 の二値分類を転移学習で行う。

入力: data/raw/{shiba, not_shiba}/
出力: models/shiba_classifier.pth

手法: MobileNetV2 (ImageNet事前学習) の分類ヘッドのみ付け替え
      --unfreeze N で特徴抽出の最後N層も微調整可能

使い方:
  python3 src/models/train_model.py
  python3 src/models/train_model.py --epochs 30 --lr 0.0005 --unfreeze 3
"""

import argparse
import json
import os
import random
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"

RANDOM_SEED = 42
LABELS = ["shiba", "not_shiba"]


# ── データ前処理 ──────────────────────────────────────────────────────

def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def make_data_dir() -> Path:
    """data/raw/ から shiba/ と not_shiba/ だけをシンボリックリンクで参照する。"""
    tmp = Path(tempfile.mkdtemp())
    for label in LABELS:
        src = RAW_DIR / label
        if src.exists():
            os.symlink(src, tmp / label)
    return tmp


def build_splits(data_dir: Path, train_tf, val_tf, val_ratio=0.15, test_ratio=0.15):
    """ImageFolder をインデックスベースで train/val/test に分割する。"""
    full_ds = datasets.ImageFolder(data_dir, transform=train_tf)
    class_names = full_ds.classes

    random.seed(RANDOM_SEED)
    indices_by_class = {}
    for idx, (_, label) in enumerate(full_ds.samples):
        indices_by_class.setdefault(label, []).append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for label, idxs in indices_by_class.items():
        random.shuffle(idxs)
        n = len(idxs)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        test_idx.extend(idxs[:n_test])
        val_idx.extend(idxs[n_test : n_test + n_val])
        train_idx.extend(idxs[n_test + n_val :])

    val_ds = datasets.ImageFolder(data_dir, transform=val_tf)

    return Subset(full_ds, train_idx), Subset(val_ds, val_idx), Subset(val_ds, test_idx), class_names


# ── モデル構築 ────────────────────────────────────────────────────────

def build_model(num_classes: int = 2, unfreeze_last: int = 0) -> nn.Module:
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")

    for param in model.parameters():
        param.requires_grad = False

    if unfreeze_last > 0:
        layers = list(model.features.children())
        for layer in layers[-unfreeze_last:]:
            for param in layer.parameters():
                param.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes),
    )
    return model


# ── 学習・評価 ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


# ── メイン ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="柴犬分類モデル学習")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--unfreeze", type=int, default=0, help="微調整する特徴抽出の最終層数")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        args.device if args.device != "auto" else "cpu"
    )
    print(f"Device: {device}\n")

    # データ準備 (shiba/ と not_shiba/ のみ参照)
    data_dir = make_data_dir()
    train_tf, val_tf = get_transforms()
    train_set, val_set, test_set, class_names = build_splits(data_dir, train_tf, val_tf)

    print(f"Classes: {class_names}")
    print(f"Train: {len(train_set)}枚 / Val: {len(val_set)}枚 / Test: {len(test_set)}枚\n")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # モデル
    model = build_model(num_classes=len(class_names), unfreeze_last=args.unfreeze).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}\n")

    # 学習クラスの不均衡補正 (shiba ~42枚 vs not_shiba ~300枚)
    class_counts = [0] * len(class_names)
    for _, label in train_set.dataset.samples:
        class_counts[label] += 1
    weights = [1.0 / c for c in class_counts]
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 学習ループ
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    header = f"{'Epoch':>5} | {'Train Loss':>10} {'Train Acc':>10} | {'Val Loss':>10} {'Val Acc':>10}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        for k, v in [("train_loss", tr_loss), ("train_acc", tr_acc),
                      ("val_loss", vl_loss), ("val_acc", vl_acc)]:
            history[k].append(v)

        marker = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "epoch": epoch,
                "val_acc": vl_acc,
            }, MODEL_DIR / "shiba_classifier.pth")
            marker = " *"

        print(f"{epoch:5d} | {tr_loss:10.4f} {tr_acc:9.1f}% | {vl_loss:10.4f} {vl_acc:9.1f}%  ({elapsed:.1f}s){marker}")

    # テスト評価
    best_ckpt = torch.load(MODEL_DIR / "shiba_classifier.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"\n{'='*50}")
    print(f"Best val acc : {best_val_acc:.1f}% (epoch {best_ckpt['epoch']})")
    print(f"Test acc     : {test_acc:.1f}%")
    print(f"Model saved  : models/shiba_classifier.pth")

    # 履歴保存
    with open(MODEL_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
