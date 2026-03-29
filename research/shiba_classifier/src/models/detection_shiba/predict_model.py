"""
学習済みモデルで画像を柴犬かどうか判定する。

使い方:
  python src/models/predict_model.py image1.jpg image2.png
  python src/models/predict_model.py --dir path/to/images/
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "shiba_classifier.pth"

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(model_path: Path = MODEL_PATH, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]

    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(in_features, len(class_names)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    return model, class_names, device


def predict(image_path: str, model, class_names, device) -> dict:
    img = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).squeeze()
    predicted = class_names[probs.argmax().item()]
    return {
        "file": str(image_path),
        "prediction": predicted,
        "confidence": probs.max().item(),
        "probabilities": {n: probs[i].item() for i, n in enumerate(class_names)},
    }


def main():
    parser = argparse.ArgumentParser(description="柴犬判定")
    parser.add_argument("images", nargs="*", help="判定する画像ファイル")
    parser.add_argument("--dir", type=str, help="画像フォルダを指定")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH))
    args = parser.parse_args()

    paths = [Path(p) for p in args.images]
    if args.dir:
        d = Path(args.dir)
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            paths.extend(d.glob(ext))

    if not paths:
        parser.print_help()
        return

    model, class_names, device = load_model(Path(args.model))
    print(f"Classes: {class_names}\n")

    for p in sorted(paths):
        r = predict(p, model, class_names, device)
        mark = "柴犬!" if r["prediction"] == "shiba" else "other"
        print(f"  {mark:6s} {r['confidence']*100:5.1f}%  {p.name}")


if __name__ == "__main__":
    main()
