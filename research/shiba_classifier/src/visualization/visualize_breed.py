"""
学習済み犬種分類モデルを使ったマスキング可視化

- 犬 (COCO class 18) → 犬種名 + Top-3 表示
- 犬以外 (人間等) → グレーマスク + COCOクラス名
- 出力はフォルダごとに保存

使い方:
  # 単体画像
  python3 src/visualization/visualize_breed.py 138169.jpg 138181.jpg

  # val データ全体 (犬種フォルダごとに出力)
  python3 src/visualization/visualize_breed.py --dir data/processed/val/

  # 出力先指定
  python3 src/visualization/visualize_breed.py --dir data/processed/val/ --out reports/figures/breed/
"""

import argparse
import colorsys
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BREED_MODEL_PATH = PROJECT_ROOT / "models" / "breed_classifier.pth"
OUT_DIR = PROJECT_ROOT / "reports" / "figures" / "breed"

MASK_THRESHOLD = 0.5
DETECT_THRESHOLD = 0.7
DOG_CLASS_ID = 18
COLOR_OTHER = (128, 128, 128)

COCO_NAMES = {
    1: "Person", 2: "Bicycle", 3: "Car", 4: "Motorcycle",
    15: "Bird", 16: "Cat", 18: "Dog", 19: "Horse", 20: "Sheep",
    21: "Cow", 22: "Elephant", 23: "Bear", 24: "Zebra", 25: "Giraffe",
}

CLASSIFY_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def clean_name(raw):
    return (raw.split("-", 1)[1] if "-" in raw else raw).replace("_", " ").title()


def breed_color(name):
    h = int(hashlib.md5(name.encode()).hexdigest()[:8], 16) % 360
    r, g, b = colorsys.hsv_to_rgb(h / 360, 0.7, 0.85)
    return (int(r * 255), int(g * 255), int(b * 255))


def load_detector(device):
    return maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).to(device).eval()


def load_breed_classifier(device):
    ckpt = torch.load(BREED_MODEL_PATH, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    model = models.mobilenet_v2(weights=None)
    inf = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(inf, len(class_names)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), class_names


def detect_objects(detector, image, device):
    img_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = detector(img_tensor)[0]
    objects = []
    for i in range(len(preds["labels"])):
        if preds["scores"][i].item() >= DETECT_THRESHOLD:
            objects.append({
                "mask": preds["masks"][i, 0].cpu().numpy(),
                "box": preds["boxes"][i].cpu().numpy().astype(int),
                "coco_id": preds["labels"][i].item(),
            })
    return objects


def classify_breed(classifier, class_names, image, box, device):
    x1, y1, x2, y2 = box
    tensor = CLASSIFY_TF(image.crop((x1, y1, x2, y2))).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(classifier(tensor), dim=1).squeeze()
    top_probs, top_idx = probs.topk(3)
    return [(class_names[top_idx[j].item()], top_probs[j].item()) for j in range(3)]


def process_image(img_path, detector, classifier, class_names, device, out_dir, subfolder=None):
    image = Image.open(img_path).convert("RGB")
    objects = detect_objects(detector, image, device)

    if not objects:
        return

    img_array = np.array(image).copy()
    results = []

    for obj in objects:
        if obj["coco_id"] != DOG_CLASS_ID:
            continue
        top3 = classify_breed(classifier, class_names, image, obj["box"], device)
        results.append({**obj, "top1": top3[0][0], "conf": top3[0][1]})
        color = breed_color(top3[0][0])
        binary_mask = obj["mask"] > MASK_THRESHOLD
        for c in range(3):
            img_array[binary_mask, c] = (img_array[binary_mask, c] * 0.5 + color[c] * 0.5).astype(np.uint8)

    if not results:
        return

    result_img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(result_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    for r in results:
        x1, y1, x2, y2 = r["box"]
        color = breed_color(r["top1"])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{clean_name(r['top1'])} {r['conf']*100:.0f}%"
        bbox = draw.textbbox((x1, y1), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        label_y = max(y1 - th - 8, 0)
        draw.rectangle([x1, label_y, x1 + tw + 8, label_y + th + 6], fill=color)
        draw.text((x1 + 4, label_y + 2), label, fill="white", font=font)

    save_dir = out_dir / subfolder if subfolder else out_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{img_path.stem}.jpg"
    result_img.save(out_path, quality=95)

    for r in results:
        print(f"  {img_path.name}: {clean_name(r['top1'])} ({r['conf']*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="犬種マスキング可視化")
    parser.add_argument("images", nargs="*", help="画像ファイル")
    parser.add_argument("--dir", type=str, help="画像フォルダ")
    parser.add_argument("--out", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("モデル読み込み中...")
    detector = load_detector(device)
    classifier, class_names = load_breed_classifier(device)
    print(f"犬種数: {len(class_names)}\n")

    out_dir = Path(args.out)

    for img_file in args.images:
        process_image(Path(img_file), detector, classifier, class_names, device, out_dir)

    if args.dir:
        dir_path = Path(args.dir)
        subdirs = [d for d in sorted(dir_path.iterdir()) if d.is_dir()]
        total = 0
        if subdirs:
            for subdir in subdirs:
                imgs = sorted([f for f in subdir.glob("*.*") if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
                for img_path in imgs:
                    process_image(img_path, detector, classifier, class_names, device, out_dir, subfolder=subdir.name)
                    total += 1
                if total % 100 == 0 and total > 0:
                    print(f"  {total}枚処理済み...")
        else:
            imgs = sorted([f for f in dir_path.glob("*.*") if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
            for img_path in imgs:
                process_image(img_path, detector, classifier, class_names, device, out_dir)
                total += 1

        print(f"\n合計 {total}枚処理")

    print(f"出力先: {out_dir}/")


if __name__ == "__main__":
    main()
