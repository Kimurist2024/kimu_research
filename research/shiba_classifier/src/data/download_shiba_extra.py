"""
複数の無料ソースから柴犬の実写画像を収集する。
AI生成・イラスト・加工画像は除外する。

ソース:
  1. Wikimedia Commons API (認証不要)
  2. Dog CEO API (認証不要, 既存分はスキップ)
"""

import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Tuple

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "shiba"

# AI生成・イラスト・加工画像を除外するためのキーワード
REJECT_KEYWORDS = [
    "dalle", "dall-e", "ai_generated", "ai-generated", "artificial",
    "generated", "stable_diffusion", "midjourney", "illustration",
    "cartoon", "drawing", "painting", "art", "anime", "icon",
    "logo", "clipart", "vector", "render", "3d", "cgi",
    "meme", "doge", "cryptocurrency", "coin", "nft",
    "statue", "figurine", "plush", "toy", "stuffed",
]


def is_rejected(url: str, title: str = "") -> bool:
    """URL/タイトルに加工画像を示すキーワードが含まれるかチェック。"""
    text = (url + " " + title).lower()
    return any(kw in text for kw in REJECT_KEYWORDS)


def download_file(url: str, dest: Path, timeout: int = 30) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ShibaClassifier/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            if len(data) < 5_000:
                return False
            dest.write_bytes(data)
            return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
        return False


# ── Source 1: Wikimedia Commons ──────────────────────────────────────

def fetch_wikimedia(query: str, limit: int = 50) -> List[Tuple[str, str]]:
    """Wikimedia Commons から画像URLとタイトルを取得。"""
    params = (
        f"action=query&generator=search&gsrsearch={urllib.request.quote(query)}"
        f"&gsrnamespace=6&gsrlimit={limit}"
        f"&prop=imageinfo&iiprop=url|size|mime"
        f"&iiurlwidth=800&format=json"
    )
    url = f"https://commons.wikimedia.org/w/api.php?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "ShibaClassifier/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    results = []
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        title = page.get("title", "")
        for ii in page.get("imageinfo", []):
            mime = ii.get("mime", "")
            if not mime.startswith("image/jpeg") and not mime.startswith("image/png"):
                continue
            if ii.get("width", 0) < 200:
                continue
            img_url = ii.get("thumburl") or ii.get("url", "")
            if img_url:
                results.append((img_url, title))
    return results


def collect_wikimedia() -> List[Tuple[str, str]]:
    """複数クエリでWikimediaから柴犬画像を収集。"""
    queries = [
        "shiba inu dog photo",
        "shiba inu puppy",
        "shiba ken japanese dog",
        "柴犬",
    ]
    all_urls = {}
    for q in queries:
        print(f"  Wikimedia検索: '{q}'")
        try:
            results = fetch_wikimedia(q, limit=50)
            for url, title in results:
                if url not in all_urls:
                    all_urls[url] = title
            time.sleep(1)
        except Exception as e:
            print(f"    エラー: {e}")
    return list(all_urls.items())


# ── Source 2: Dog CEO API ────────────────────────────────────────────

def fetch_dog_ceo() -> List[Tuple[str, str]]:
    url = "https://dog.ceo/api/breed/shiba/images"
    req = urllib.request.Request(url, headers={"User-Agent": "ShibaClassifier/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    if data.get("status") == "success":
        return [(u, "") for u in data["message"]]
    return []


# ── Main ─────────────────────────────────────────────────────────────

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing = set(f.name for f in RAW_DIR.iterdir() if f.is_file())
    existing_count = len(existing)
    print(f"既存画像: {existing_count}枚\n")

    # 全ソースからURL収集
    all_candidates = []

    print("[1/2] Dog CEO API...")
    all_candidates.extend(fetch_dog_ceo())
    print(f"  → {len(all_candidates)}件")

    print("[2/2] Wikimedia Commons...")
    wiki_results = collect_wikimedia()
    all_candidates.extend(wiki_results)
    print(f"  → 累計 {len(all_candidates)}件\n")

    # 重複URL除去
    seen_urls = set()
    unique = []
    for url, title in all_candidates:
        if url not in seen_urls:
            seen_urls.add(url)
            unique.append((url, title))

    # フィルタリング: AI生成・加工画像を除外
    filtered = []
    rejected_count = 0
    for url, title in unique:
        if is_rejected(url, title):
            rejected_count += 1
        else:
            filtered.append((url, title))

    print(f"候補: {len(unique)}件 → フィルタ後: {len(filtered)}件 (除外: {rejected_count}件)")
    print()

    # ダウンロード
    idx = existing_count
    downloaded = 0
    failed = 0

    for url, title in filtered:
        idx += 1
        ext = ".jpg"
        if ".png" in url.lower():
            ext = ".png"
        filename = f"shiba_{idx:04d}{ext}"
        dest = RAW_DIR / filename

        if dest.exists():
            continue

        ok = download_file(url, dest)
        if ok:
            downloaded += 1
            if downloaded % 10 == 0 or downloaded <= 5:
                print(f"  [{downloaded}] {filename}")
        else:
            failed += 1
            idx -= 1  # 番号を戻す

        time.sleep(0.5)

    total = len(list(RAW_DIR.glob("*.*")))
    print(f"\n完了: 新規{downloaded}枚DL, {failed}枚失敗")
    print(f"data/raw/shiba/ 合計: {total}枚")

    if total < 150:
        shortage = 150 - total
        print(f"\nまだ{shortage}枚不足しています。")
        print("手動で追加する場合: data/raw/shiba/ に jpg/png を置いてください。")


if __name__ == "__main__":
    main()
