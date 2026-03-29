"""
犬種分類モデル（121クラス: Stanford Dogs 120犬種 + 柴犬）

入力: data/raw/somedog_image/  (120犬種)
      data/raw/shiba/          (柴犬)
出力: models/breed_classifier.pth

手法: MobileNetV2 (ImageNet事前学習) の分類ヘッド + 最終N層を微調整

使い方:
  python3 src/models/breed_classifier/train_model.py
  python3 src/models/breed_classifier/train_model.py --epochs 30 --unfreeze 5
"""

import argparse
import json
import os
import random
import shutil
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"

RANDOM_SEED = 42


# ── データ前処理 ──────────────────────────────────────────────────────

def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
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
    """somedog_image/ の全犬種 + shiba/ をシンボリックリンクで統合する。"""
    tmp = Path(tempfile.mkdtemp())

    # Stanford Dogs 120犬種
    somedog = RAW_DIR / "somedog_image"
    for breed_dir in sorted(somedog.iterdir()):
        if breed_dir.is_dir():
            os.symlink(breed_dir, tmp / breed_dir.name)

    # 柴犬を追加 (ImageFolder互換の名前で)
    shiba_src = RAW_DIR / "shiba"
    if shiba_src.exists():
        os.symlink(shiba_src, tmp / "shiba_inu")

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

def build_model(num_classes: int, unfreeze_last: int = 3) -> nn.Module:
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")

    # まず全て凍結
    for param in model.parameters():
        param.requires_grad = False

    # 最後N層を解凍 (121クラスでは特徴抽出側も微調整が必要)
    if unfreeze_last > 0:
        layers = list(model.features.children())
        for layer in layers[-unfreeze_last:]:
            for param in layer.parameters():
                param.requires_grad = True

    # 分類ヘッドを付け替え
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
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
    top5_correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        # Top-5 accuracy
        _, top5_pred = outputs.topk(5, dim=1)
        top5_correct += top5_pred.eq(labels.unsqueeze(1)).any(1).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total, 100.0 * top5_correct / total


# ── メイン ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="犬種分類モデル学習 (121クラス)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--unfreeze", type=int, default=3, help="微調整する特徴抽出の最終層数")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        args.device if args.device != "auto" else "cpu"
    )
    print(f"Device: {device}\n")

    # データ準備 (data/processed/{train,val,test}/ を直接読む)
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    train_tf, val_tf = get_transforms()

    train_set = datasets.ImageFolder(PROCESSED_DIR / "train", transform=train_tf)
    test_set = datasets.ImageFolder(PROCESSED_DIR / "test", transform=val_tf)
    class_names = train_set.classes
    num_classes = len(class_names)

    print(f"Classes: {num_classes}犬種")
    print(f"Train: {len(train_set)}枚 / Test: {len(test_set)}枚")
    print(f"Val: {PROCESSED_DIR / 'val'} (学習後にマスク画像出力用)\n")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # モデル
    model = build_model(num_classes=num_classes, unfreeze_last=args.unfreeze).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total_params:,} params\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 学習ループ (train で学習、test で評価)
    best_test_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], "test_top5": []}

    header = f"{'Epoch':>5} | {'Tr Loss':>8} {'Tr Acc':>7} | {'Ts Loss':>8} {'Ts Acc':>7} {'Top-5':>6}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        ts_loss, ts_acc, ts_top5 = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(ts_loss)
        history["test_acc"].append(ts_acc)
        history["test_top5"].append(ts_top5)

        marker = ""
        if ts_acc > best_test_acc:
            best_test_acc = ts_acc
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "num_classes": num_classes,
                "epoch": epoch,
                "test_acc": ts_acc,
                "test_top5": ts_top5,
            }, MODEL_DIR / "breed_classifier.pth")
            marker = " *"

        print(f"{epoch:5d} | {tr_loss:8.4f} {tr_acc:6.1f}% | {ts_loss:8.4f} {ts_acc:6.1f}% {ts_top5:5.1f}%  ({elapsed:.1f}s){marker}")

    # ベストモデル読み込み
    best_ckpt = torch.load(MODEL_DIR / "breed_classifier.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    print(f"\n{'='*60}")
    print(f"Best test acc : {best_test_acc:.1f}% (epoch {best_ckpt['epoch']})")
    print(f"Model saved   : models/breed_classifier.pth")
    print(f"Classes       : {num_classes}")

    # 履歴保存
    with open(MODEL_DIR / "breed_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── Val データでマスク画像を出力 ──────────────────────────────────
    print(f"\n{'='*60}")
    print("Val データでマスク画像を生成中...")

    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    import colorsys
    import hashlib

    FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "breed"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Mask R-CNN 読み込み
    detector = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).to(device).eval()

    def clean_name(raw):
        return (raw.split("-", 1)[1] if "-" in raw else raw).replace("_", " ").title()

    def breed_color(name):
        h = int(hashlib.md5(name.encode()).hexdigest()[:8], 16) % 360
        r, g, b = colorsys.hsv_to_rgb(h / 360, 0.7, 0.85)
        return (int(r * 255), int(g * 255), int(b * 255))

    classify_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # val の画像パスを収集
    val_dir = PROCESSED_DIR / "val"
    val_images = []
    for breed_dir in sorted(val_dir.iterdir()):
        if breed_dir.is_dir():
            for img_path in sorted(breed_dir.glob("*.*")):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    val_images.append(img_path)

    saved = 0
    for img_path in val_images:
        image = Image.open(img_path).convert("RGB")
        img_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

        # 物体検出
        with torch.no_grad():
            preds = detector(img_tensor)[0]

        # 検出結果を犬(COCO class 18)とそれ以外に分ける
        # COCO: 1=person, 16=cat, 18=dog
        COCO_NAMES = {1: "Person", 2: "Bicycle", 3: "Car", 16: "Cat", 18: "Dog"}
        DOG_CLASS_ID = 18
        COLOR_OTHER = (128, 128, 128)  # 犬以外はグレー

        detections = []
        for i in range(len(preds["labels"])):
            if preds["scores"][i].item() >= 0.7:
                mask = preds["masks"][i, 0].cpu().numpy()
                box = preds["boxes"][i].cpu().numpy().astype(int)
                label_id = preds["labels"][i].item()
                detections.append({"mask": mask, "box": box, "coco_id": label_id})

        if not detections:
            continue

        # 犬のみ犬種分類、それ以外はCOCOクラス名を表示
        img_array = np.array(image).copy()
        results = []
        for det in detections:
            if det["coco_id"] == DOG_CLASS_ID:
                # 犬 → 犬種分類
                x1, y1, x2, y2 = det["box"]
                cropped = image.crop((x1, y1, x2, y2))
                tensor = classify_tf(cropped).unsqueeze(0).to(device)
                with torch.no_grad():
                    probs = torch.nn.functional.softmax(model(tensor), dim=1).squeeze()
                top3_probs, top3_idx = probs.topk(3)
                top1_name = class_names[top3_idx[0].item()]
                top1_conf = top3_probs[0].item()
                top3 = [(class_names[top3_idx[j].item()], top3_probs[j].item()) for j in range(3)]
                results.append({**det, "is_dog": True, "top1": top1_name, "conf": top1_conf, "top3": top3})

                color = breed_color(top1_name)
                binary_mask = det["mask"] > 0.5
                for c in range(3):
                    img_array[binary_mask, c] = (
                        img_array[binary_mask, c] * 0.5 + color[c] * 0.5
                    ).astype(np.uint8)
            else:
                # 犬以外 → COCOクラス名のみ
                coco_name = COCO_NAMES.get(det["coco_id"], f"Object({det['coco_id']})")
                results.append({**det, "is_dog": False, "coco_name": coco_name})

                binary_mask = det["mask"] > 0.5
                for c in range(3):
                    img_array[binary_mask, c] = (
                        img_array[binary_mask, c] * 0.5 + COLOR_OTHER[c] * 0.5
                    ).astype(np.uint8)

        # ラベル描画
        result_img = Image.fromarray(img_array)
        draw = ImageDraw.Draw(result_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
            font_small = font

        for r in results:
            x1, y1, x2, y2 = r["box"]

            if r["is_dog"]:
                # 犬 → 犬種名 + Top-3
                color = breed_color(r["top1"])
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                label = f"{clean_name(r['top1'])} {r['conf']*100:.0f}%"
                bbox = draw.textbbox((x1, y1), label, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                label_y = max(y1 - th - 8, 0)
                draw.rectangle([x1, label_y, x1 + tw + 8, label_y + th + 6], fill=color)
                draw.text((x1 + 4, label_y + 2), label, fill="white", font=font)

                sub_y = label_y + th + 8
                for rank, (bname, bprob) in enumerate(r["top3"][1:], 2):
                    sub_label = f"{rank}. {clean_name(bname)} {bprob*100:.0f}%"
                    sb = draw.textbbox((x1, sub_y), sub_label, font=font_small)
                    stw, sth = sb[2] - sb[0], sb[3] - sb[1]
                    draw.rectangle([x1, sub_y, x1 + stw + 6, sub_y + sth + 4], fill=(0, 0, 0, 180))
                    draw.text((x1 + 3, sub_y + 1), sub_label, fill="white", font=font_small)
                    sub_y += sth + 6
            else:
                # 犬以外 → COCOクラス名のみ (グレー)
                draw.rectangle([x1, y1, x2, y2], outline=COLOR_OTHER, width=2)
                label = r["coco_name"]
                bbox = draw.textbbox((x1, y1), label, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                label_y = max(y1 - th - 8, 0)
                draw.rectangle([x1, label_y, x1 + tw + 8, label_y + th + 6], fill=COLOR_OTHER)
                draw.text((x1 + 4, label_y + 2), label, fill="white", font=font)

        # 犬種フォルダごとに保存
        breed_fig_dir = FIGURES_DIR / img_path.parent.name
        breed_fig_dir.mkdir(parents=True, exist_ok=True)
        out_path = breed_fig_dir / f"{img_path.stem}.jpg"
        result_img.save(out_path, quality=95)
        saved += 1

        if saved % 100 == 0:
            print(f"  {saved}枚保存済み...")

    print(f"\nマスク画像 {saved}枚を保存しました")
    print(f"出力先: {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
