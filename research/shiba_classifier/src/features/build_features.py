"""
data/raw/ から data/processed/{train,val,test}/ に犬種ごと分割コピーする。

入力:
  data/raw/somedog_image/ (120犬種)
  data/raw/shiba/         (柴犬)

出力:
  data/processed/train/{犬種名}/
  data/processed/val/{犬種名}/
  data/processed/test/{犬種名}/

使い方:
  python3 src/features/build_features.py
  python3 src/features/build_features.py --clean
"""

import argparse
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RANDOM_SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}


def collect_breeds():
    """全犬種のフォルダと画像リストを返す。"""
    breeds = {}

    # Stanford Dogs 120犬種
    somedog = RAW_DIR / "somedog_image"
    if somedog.exists():
        for breed_dir in sorted(somedog.iterdir()):
            if breed_dir.is_dir():
                imgs = [f for f in breed_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
                if imgs:
                    breeds[breed_dir.name] = sorted(imgs)

    # 柴犬
    shiba = RAW_DIR / "shiba"
    if shiba.exists():
        imgs = [f for f in shiba.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        if imgs:
            breeds["shiba_inu"] = sorted(imgs)

    return breeds


def split_and_copy(breeds: dict):
    """各犬種を train/val/test に分割してコピー。"""
    random.seed(RANDOM_SEED)
    total_counts = {"train": 0, "val": 0, "test": 0}

    for breed_name, imgs in breeds.items():
        random.shuffle(imgs)
        n = len(imgs)
        n_test = int(n * SPLITS["test"])
        n_val = int(n * SPLITS["val"])

        splits = {
            "test": imgs[:n_test],
            "val": imgs[n_test : n_test + n_val],
            "train": imgs[n_test + n_val :],
        }

        for split_name, split_imgs in splits.items():
            dest_dir = PROCESSED_DIR / split_name / breed_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img in split_imgs:
                shutil.copy2(img, dest_dir / img.name)
            total_counts[split_name] += len(split_imgs)

    return total_counts


def main():
    parser = argparse.ArgumentParser(description="犬種データセット分割")
    parser.add_argument("--clean", action="store_true", help="既存のprocessed/を削除して再構築")
    args = parser.parse_args()

    if args.clean and PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
        print("data/processed/ をクリーンアップ\n")

    breeds = collect_breeds()
    print(f"犬種数: {len(breeds)}")
    print(f"合計画像数: {sum(len(v) for v in breeds.values())}枚\n")

    counts = split_and_copy(breeds)

    print("分割結果:")
    for split, count in counts.items():
        breed_count = len(list((PROCESSED_DIR / split).iterdir()))
        print(f"  {split}: {count}枚 ({breed_count}犬種)")

    print(f"\n出力先: {PROCESSED_DIR}/")


if __name__ == "__main__":
    main()
