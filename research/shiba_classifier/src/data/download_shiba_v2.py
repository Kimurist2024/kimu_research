"""
柴犬の実写画像を150枚以上集めるスクリプト v2

ソース:
  1. Dog CEO API
  2. Wikimedia Commons (フルサイズURL使用)
  3. Wikipedia記事内の画像
"""

import json
import os
import hashlib
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Set, Tuple

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "shiba"

REJECT_KEYWORDS = [
    "dalle", "dall-e", "ai_generated", "ai-generated",
    "stable_diffusion", "midjourney", "illustration",
    "cartoon", "drawing", "painting", "anime", "icon",
    "logo", "clipart", "vector", "render", "3d", "cgi",
    "meme", "doge", "cryptocurrency", "coin", "nft",
    "statue", "figurine", "plush", "toy", "stuffed",
    "diagram", "map", "chart", "graph", "flag", "emblem",
    "kanji", "calligraphy", "text", "screenshot",
]

HEADERS = {"User-Agent": "ShibaClassifier/1.0 (academic research; mailto:shiba@example.com)"}


def is_rejected(url, title=""):
    text = (url + " " + title).lower()
    return any(kw in text for kw in REJECT_KEYWORDS)


def url_get(url, timeout=30):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download_image(url, dest, timeout=30):
    try:
        data = url_get(url, timeout)
        if len(data) < 5_000:
            return False
        # JPEG/PNG magic bytes check
        if not (data[:2] == b'\xff\xd8' or data[:4] == b'\x89PNG'):
            return False
        dest.write_bytes(data)
        return True
    except Exception:
        return False


def content_hash(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


# ── Dog CEO API ──

def fetch_dog_ceo():
    data = json.loads(url_get("https://dog.ceo/api/breed/shiba/images"))
    return data.get("message", []) if data.get("status") == "success" else []


# ── Wikimedia Commons (フルサイズ) ──

def fetch_wikimedia_full(query, limit=50):
    """imageinfo で元画像URLを取得。"""
    params = (
        f"action=query&generator=search"
        f"&gsrsearch={urllib.request.quote(query)}"
        f"&gsrnamespace=6&gsrlimit={limit}"
        f"&prop=imageinfo&iiprop=url|mime|size"
        f"&format=json"
    )
    url = f"https://commons.wikimedia.org/w/api.php?{params}"
    data = json.loads(url_get(url))
    results = []
    for page in data.get("query", {}).get("pages", {}).values():
        title = page.get("title", "")
        for ii in page.get("imageinfo", []):
            mime = ii.get("mime", "")
            if mime not in ("image/jpeg", "image/png"):
                continue
            w = ii.get("width", 0)
            if w < 200:
                continue
            img_url = ii.get("url", "")
            if img_url:
                results.append((img_url, title))
    return results


def fetch_wikimedia_category(category, limit=50):
    """カテゴリ内の画像を取得。"""
    params = (
        f"action=query&generator=categorymembers"
        f"&gcmtitle={urllib.request.quote(category)}"
        f"&gcmtype=file&gcmlimit={limit}"
        f"&prop=imageinfo&iiprop=url|mime|size"
        f"&format=json"
    )
    url = f"https://commons.wikimedia.org/w/api.php?{params}"
    data = json.loads(url_get(url))
    results = []
    for page in data.get("query", {}).get("pages", {}).values():
        title = page.get("title", "")
        for ii in page.get("imageinfo", []):
            mime = ii.get("mime", "")
            if mime not in ("image/jpeg", "image/png"):
                continue
            img_url = ii.get("url", "")
            if img_url:
                results.append((img_url, title))
    return results


# ── Wikipedia 記事内画像 ──

def fetch_wikipedia_article_images(titles):
    """Wikipedia記事ページ内の画像を取得。"""
    results = []
    for article in titles:
        params = (
            f"action=query&titles={urllib.request.quote(article)}"
            f"&prop=images&imlimit=50&format=json"
        )
        url = f"https://en.wikipedia.org/w/api.php?{params}"
        try:
            data = json.loads(url_get(url))
            image_titles = []
            for page in data.get("query", {}).get("pages", {}).values():
                for img in page.get("images", []):
                    t = img.get("title", "")
                    if t.lower().endswith((".jpg", ".jpeg", ".png")):
                        image_titles.append(t)

            # 各画像のURLを取得
            for img_title in image_titles:
                info_params = (
                    f"action=query&titles={urllib.request.quote(img_title)}"
                    f"&prop=imageinfo&iiprop=url|mime|size&format=json"
                )
                info_url = f"https://en.wikipedia.org/w/api.php?{info_params}"
                info_data = json.loads(url_get(info_url))
                for p in info_data.get("query", {}).get("pages", {}).values():
                    for ii in p.get("imageinfo", []):
                        if ii.get("mime", "") in ("image/jpeg", "image/png"):
                            results.append((ii["url"], img_title))
            time.sleep(0.5)
        except Exception as e:
            print(f"    Wikipedia '{article}' エラー: {e}")
    return results


# ── Main ──

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 既存ファイルのハッシュ (重複防止)
    existing_hashes = set()
    existing_files = list(RAW_DIR.glob("*.*"))
    for f in existing_files:
        try:
            existing_hashes.add(content_hash(f))
        except Exception:
            pass
    print(f"既存画像: {len(existing_files)}枚\n")

    all_urls = {}  # url -> title

    # 1. Dog CEO
    print("[1/4] Dog CEO API...")
    for u in fetch_dog_ceo():
        all_urls[u] = ""
    print(f"  → {len(all_urls)}件")

    # 2. Wikimedia 検索
    print("[2/4] Wikimedia Commons 検索...")
    for q in ["shiba inu dog", "shiba inu puppy", "shiba ken", "柴犬 dog"]:
        print(f"  検索: '{q}'")
        try:
            for url, title in fetch_wikimedia_full(q, limit=50):
                if url not in all_urls:
                    all_urls[url] = title
            time.sleep(1)
        except Exception as e:
            print(f"    エラー: {e}")
    print(f"  → 累計 {len(all_urls)}件")

    # 3. Wikimedia カテゴリ
    print("[3/4] Wikimedia Commons カテゴリ...")
    categories = [
        "Category:Shiba Inu",
        "Category:Shiba inu",
        "Category:Shiba (dog)",
    ]
    for cat in categories:
        print(f"  カテゴリ: '{cat}'")
        try:
            for url, title in fetch_wikimedia_category(cat, limit=50):
                if url not in all_urls:
                    all_urls[url] = title
            time.sleep(1)
        except Exception as e:
            print(f"    エラー: {e}")
    print(f"  → 累計 {len(all_urls)}件")

    # 4. Wikipedia 記事
    print("[4/4] Wikipedia 記事内画像...")
    articles = ["Shiba Inu", "Shiba inu (disambiguation)", "Japanese dog breeds"]
    wp_results = fetch_wikipedia_article_images(articles)
    for url, title in wp_results:
        if url not in all_urls:
            all_urls[url] = title
    print(f"  → 累計 {len(all_urls)}件\n")

    # フィルタ
    filtered = [(u, t) for u, t in all_urls.items() if not is_rejected(u, t)]
    rejected = len(all_urls) - len(filtered)
    print(f"フィルタ: {len(all_urls)}件 → {len(filtered)}件 (除外: {rejected}件)\n")

    # ダウンロード
    idx = len(existing_files)
    downloaded = 0
    skipped_dup = 0
    failed = 0

    for url, title in filtered:
        idx += 1
        ext = ".png" if ".png" in url.lower().split("?")[0] else ".jpg"
        filename = f"shiba_{idx:04d}{ext}"
        dest = RAW_DIR / filename

        ok = download_image(url, dest)
        if ok:
            h = content_hash(dest)
            if h in existing_hashes:
                dest.unlink()
                skipped_dup += 1
                idx -= 1
                continue
            existing_hashes.add(h)
            downloaded += 1
            if downloaded % 10 == 0 or downloaded <= 3:
                print(f"  [{downloaded}] {filename}")
        else:
            if dest.exists():
                dest.unlink()
            failed += 1
            idx -= 1

        time.sleep(0.3)

    total = len(list(RAW_DIR.glob("*.*")))
    print(f"\n===== 結果 =====")
    print(f"新規DL: {downloaded}枚")
    print(f"重複スキップ: {skipped_dup}枚")
    print(f"失敗: {failed}枚")
    print(f"data/raw/shiba/ 合計: {total}枚")

    if total >= 150:
        print(f"\n目標150枚を達成しました!")
    else:
        print(f"\nあと{150 - total}枚不足。手動で data/raw/shiba/ に追加してください。")


if __name__ == "__main__":
    main()
