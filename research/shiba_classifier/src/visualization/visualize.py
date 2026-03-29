"""
犬領域をマスキングし、shiba / not shiba の判定結果を重畳表示する。

1. Mask R-CNN で犬領域をセグメンテーション
2. 柴犬分類モデルで shiba / not_shiba を判定
3. マスク + ラベルを重ねた画像を保存

使い方:
  python3 src/visualization/visualize.py data/raw/shiba/shiba_0001.jpg
  python3 src/visualization/visualize.py --dir data/raw/shiba/ --out reports/figures/
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "shiba_classifier.pth"
OUT_DIR = PROJECT_ROOT / "reports" / "figures"

MASK_THRESHOLD = 0.5
DETECT_THRESHOLD = 0.7

# 分類用の前処理
CLASSIFY_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 色設定
COLOR_SHIBA = (0, 200, 80)       # 緑
COLOR_NOT_SHIBA = (220, 60, 60)  # 赤


def load_detector(device):
    """Mask R-CNN (犬検出用)"""
    detector = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    detector = detector.to(device).eval()
    return detector


def load_classifier(device):
    """柴犬分類モデル"""
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    model = models.mobilenet_v2(weights=None)
    inf = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(inf, len(class_names)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model, class_names


def detect_objects(detector, image: Image.Image, device):
    """画像内の全物体を検出し、マスクとバウンディングボックスを返す。"""
    img_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = detector(img_tensor)[0]

    objects = []
    for i in range(len(preds["labels"])):
        if preds["scores"][i].item() >= DETECT_THRESHOLD:
            mask = preds["masks"][i, 0].cpu().numpy()
            box = preds["boxes"][i].cpu().numpy().astype(int)
            score = preds["scores"][i].item()
            label_id = preds["labels"][i].item()
            objects.append({"mask": mask, "box": box, "det_score": score, "label_id": label_id})
    return objects


def classify_region(classifier, class_names, image: Image.Image, box, device):
    """バウンディングボックスで切り出した領域を分類。"""
    x1, y1, x2, y2 = box
    cropped = image.crop((x1, y1, x2, y2))
    tensor = CLASSIFY_TF(cropped).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(classifier(tensor), dim=1).squeeze()
    pred = class_names[probs.argmax().item()]
    conf = probs.max().item()
    return pred, conf


def draw_result(image: Image.Image, dogs_info, out_path: Path):
    """マスクとラベルを画像に重畳して保存。"""
    img_array = np.array(image).copy()

    # マスクを半透明で重畳
    for info in dogs_info:
        mask = info["mask"]
        pred = info["pred"]
        color = COLOR_SHIBA if pred == "shiba" else COLOR_NOT_SHIBA
        binary_mask = mask > MASK_THRESHOLD
        for c in range(3):
            img_array[binary_mask, c] = (
                img_array[binary_mask, c] * 0.5 + color[c] * 0.5
            ).astype(np.uint8)

    # PIL で枠線とラベル描画
    result = Image.fromarray(img_array)
    draw = ImageDraw.Draw(result)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except OSError:
        font = ImageFont.load_default()

    for info in dogs_info:
        box = info["box"]
        pred = info["pred"]
        conf = info["conf"]
        color = COLOR_SHIBA if pred == "shiba" else COLOR_NOT_SHIBA
        x1, y1, x2, y2 = box

        # バウンディングボックス
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # ラベル
        label = f"shiba {conf*100:.0f}%" if pred == "shiba" else f"not shiba {conf*100:.0f}%"
        bbox = draw.textbbox((x1, y1), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        label_y = max(y1 - th - 8, 0)
        draw.rectangle([x1, label_y, x1 + tw + 8, label_y + th + 6], fill=color)
        draw.text((x1 + 4, label_y + 2), label, fill="white", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path, quality=95)
    return result


def process_image(img_path: Path, detector, classifier, class_names, device, out_dir: Path):
    image = Image.open(img_path).convert("RGB")
    objects = detect_objects(detector, image, device)

    if not objects:
        print(f"  {img_path.name}: 何も検出されませんでした")
        return

    results = []
    for obj in objects:
        if obj["label_id"] != 18:  # 犬(COCO class 18)のみ
            continue
        pred, conf = classify_region(classifier, class_names, image, obj["box"], device)
        results.append({**obj, "pred": pred, "conf": conf})

    if not results:
        print(f"  {img_path.name}: 犬が検出されませんでした")
        return

    out_path = out_dir / f"result_{img_path.stem}.jpg"
    draw_result(image, results, out_path)

    for info in results:
        label = "shiba" if info["pred"] == "shiba" else "not shiba"
        print(f"  {img_path.name}: {label} ({info['conf']*100:.1f}%) -> {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description="犬マスキング + 柴犬判定の可視化")
    parser.add_argument("images", nargs="*", help="画像ファイル")
    parser.add_argument("--dir", type=str, help="画像フォルダ")
    parser.add_argument("--out", type=str, default=str(OUT_DIR), help="出力先フォルダ")
    args = parser.parse_args()

    paths = [Path(p) for p in args.images]
    if args.dir:
        d = Path(args.dir)
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            paths.extend(d.glob(ext))

    if not paths:
        parser.print_help()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("モデル読み込み中...")
    detector = load_detector(device)
    classifier, class_names = load_classifier(device)
    print(f"Classes: {class_names}\n")

    out_dir = Path(args.out)
    for p in sorted(paths):
        process_image(p, detector, classifier, class_names, device, out_dir)

    print(f"\n出力先: {out_dir}/")


if __name__ == "__main__":
    main()
