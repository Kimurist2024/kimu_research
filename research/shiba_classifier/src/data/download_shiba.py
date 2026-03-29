"""
柴犬画像ダウンロードスクリプト
- Dog CEO API: 実写柴犬画像を全件取得
- Stanford Dogs (Shih-Tzu隣接synset等)は柴犬を含まないため不使用
- ImageNet synset n02086240 相当の画像をFlickr経由で取得(未使用)

加工画像(イラスト・AI生成等)は除外し、実写写真のみを対象とする。
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import List

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "shiba"


def download_file(url: str, dest: Path, timeout: int = 30) -> bool:
    """1ファイルをダウンロード。失敗時はFalse。"""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ShibaClassifier/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            # 最低限の画像サイズチェック (5KB未満はサムネ/破損の可能性)
            if len(data) < 5_000:
                return False
            dest.write_bytes(data)
            return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
        return False


def fetch_dog_ceo() -> List[str]:
    """Dog CEO API から柴犬画像URLを全件取得。"""
    url = "https://dog.ceo/api/breed/shiba/images"
    req = urllib.request.Request(url, headers={"User-Agent": "ShibaClassifier/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    if data.get("status") == "success":
        return data["message"]
    return []


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # --- Dog CEO API ---
    print("Dog CEO API から柴犬画像URL取得中...")
    urls = fetch_dog_ceo()
    print(f"  取得URL数: {len(urls)}")

    downloaded = 0
    skipped = 0
    failed = 0

    for i, url in enumerate(urls):
        ext = os.path.splitext(url.split("?")[0])[-1].lower()
        if ext not in (".jpg", ".jpeg", ".png"):
            ext = ".jpg"
        filename = f"shiba_{i+1:04d}{ext}"
        dest = RAW_DIR / filename

        if dest.exists():
            skipped += 1
            continue

        ok = download_file(url, dest)
        if ok:
            downloaded += 1
            print(f"  [{downloaded}] {filename}")
        else:
            failed += 1

        time.sleep(0.3)  # rate limit 配慮

    print(f"\n完了: {downloaded}枚DL, {skipped}枚スキップ, {failed}枚失敗")

    total = len(list(RAW_DIR.glob("*.*")))
    print(f"data/raw/shiba/ 内の合計: {total}枚")

    if total < 150:
        print(f"\n注意: 目標150枚に対して{total}枚です。")
        print("追加データの候補:")
        print("  - Kaggle 'shiba inu' で検索")
        print("  - Flickr API で 'shiba inu photo' 検索")
        print("  - Google Images から手動収集")


if __name__ == "__main__":
    main()
