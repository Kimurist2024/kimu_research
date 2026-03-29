"""
柴犬の実写画像を150枚以上収集するスクリプト v3

ソース:
  1. Dog CEO API
  2. Wikimedia Commons (検索 + カテゴリ)
  3. Wikipedia記事内画像
"""

import json
import os
import hashlib
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Tuple

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "shiba"

REJECT_KEYWORDS = [
    "dalle", "dall-e", "ai_generated", "ai-generated",
    "stable_diffusion", "midjourney", "illustration",
    "cartoon", "drawing", "painting", "anime", "icon",
    "logo", "clipart", "vector", "render", "3d", "cgi",
    "meme", "doge", "cryptocurrency", "coin", "nft",
    "statue", "figurine", "plush", "toy", "stuffed",
    "diagram", "map", "chart", "graph", "flag", "emblem",
    "kanji", "calligraphy", "screenshot", "svg",
]

# Wikimedia API にはbot UA, 画像DLにはブラウザUA
API_HEADERS = {"User-Agent": "ShibaClassifierBot/1.0 (academic; bot@example.com)"}
DL_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}


def is_rejected(url: str, title: str = "") -> bool:
    text = (url + " " + title).lower()
    return any(kw in text for kw in REJECT_KEYWORDS)


def api_get(url: str) -> dict:
    req = urllib.request.Request(url, headers=API_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def download_image(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers=DL_HEADERS)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        if len(data) < 5_000:
            return False
        if not (data[:2] == b'\xff\xd8' or data[:4] == b'\x89PNG'):
            return False
        dest.write_bytes(data)
        return True
    except Exception:
        return False


def content_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# ── Dog CEO ──

def fetch_dog_ceo() -> List[str]:
    data = json.loads(
        urllib.request.urlopen(
            urllib.request.Request("https://dog.ceo/api/breed/shiba/images", headers=DL_HEADERS),
            timeout=30
        ).read()
    )
    return data.get("message", []) if data.get("status") == "success" else []


# ── Wikimedia Commons ──

def wikimedia_search(query: str, limit: int = 50) -> List[Tuple[str, str]]:
    params = (
        f"action=query&generator=search"
        f"&gsrsearch={urllib.request.quote(query)}"
        f"&gsrnamespace=6&gsrlimit={limit}"
        f"&prop=imageinfo&iiprop=url|mime|size"
        f"&format=json"
    )
    data = api_get(f"https://commons.wikimedia.org/w/api.php?{params}")
    results = []
    for page in data.get("query", {}).get("pages", {}).values():
        title = page.get("title", "")
        for ii in page.get("imageinfo", []):
            if ii.get("mime") in ("image/jpeg", "image/png") and ii.get("width", 0) >= 200:
                results.append((ii["url"], title))
    return results


def wikimedia_category(category: str, limit: int = 50) -> List[Tuple[str, str]]:
    params = (
        f"action=query&generator=categorymembers"
        f"&gcmtitle={urllib.request.quote(category)}"
        f"&gcmtype=file&gcmlimit={limit}"
        f"&prop=imageinfo&iiprop=url|mime|size"
        f"&format=json"
    )
    data = api_get(f"https://commons.wikimedia.org/w/api.php?{params}")
    results = []
    for page in data.get("query", {}).get("pages", {}).values():
        title = page.get("title", "")
        for ii in page.get("imageinfo", []):
            if ii.get("mime") in ("image/jpeg", "image/png"):
                results.append((ii["url"], title))
    return results


def wikipedia_images(articles: List[str]) -> List[Tuple[str, str]]:
    results = []
    for article in articles:
        try:
            params = (
                f"action=query&titles={urllib.request.quote(article)}"
                f"&prop=images&imlimit=50&format=json"
            )
            data = api_get(f"https://en.wikipedia.org/w/api.php?{params}")
            img_titles = []
            for page in data.get("query", {}).get("pages", {}).values():
                for img in page.get("images", []):
                    t = img.get("title", "")
                    if t.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_titles.append(t)

            for t in img_titles:
                info_params = (
                    f"action=query&titles={urllib.request.quote(t)}"
                    f"&prop=imageinfo&iiprop=url|mime&format=json"
                )
                info = api_get(f"https://en.wikipedia.org/w/api.php?{info_params}")
                for p in info.get("query", {}).get("pages", {}).values():
                    for ii in p.get("imageinfo", []):
                        if ii.get("mime") in ("image/jpeg", "image/png"):
                            results.append((ii["url"], t))
            time.sleep(0.5)
        except Exception as e:
            print(f"    Wikipedia '{article}': {e}")
    return results


# ── Japanese Wikipedia ──

def jawiki_images(articles: List[str]) -> List[Tuple[str, str]]:
    results = []
    for article in articles:
        try:
            params = (
                f"action=query&titles={urllib.request.quote(article)}"
                f"&prop=images&imlimit=50&format=json"
            )
            data = api_get(f"https://ja.wikipedia.org/w/api.php?{params}")
            img_titles = []
            for page in data.get("query", {}).get("pages", {}).values():
                for img in page.get("images", []):
                    t = img.get("title", "")
                    if t.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_titles.append(t)

            for t in img_titles:
                info_params = (
                    f"action=query&titles={urllib.request.quote(t)}"
                    f"&prop=imageinfo&iiprop=url|mime&format=json"
                )
                info = api_get(f"https://ja.wikipedia.org/w/api.php?{info_params}")
                for p in info.get("query", {}).get("pages", {}).values():
                    for ii in p.get("imageinfo", []):
                        if ii.get("mime") in ("image/jpeg", "image/png"):
                            results.append((ii["url"], t))
            time.sleep(0.5)
        except Exception as e:
            print(f"    ja.Wikipedia '{article}': {e}")
    return results


# ── Main ──

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing_hashes = set()
    existing_files = list(RAW_DIR.glob("*.*"))
    for f in existing_files:
        try:
            existing_hashes.add(content_hash(f))
        except Exception:
            pass
    print(f"既存画像: {len(existing_files)}枚\n")

    all_urls: Dict[str, str] = {}  # url -> title

    # 1. Dog CEO
    print("[1/5] Dog CEO API...")
    for u in fetch_dog_ceo():
        all_urls[u] = ""
    print(f"  {len(all_urls)}件")

    # 2. Wikimedia 検索
    print("[2/5] Wikimedia Commons 検索...")
    for q in [
        "shiba inu dog photo",
        "shiba inu puppy",
        "shiba inu",
        "shiba ken dog",
        "柴犬",
        "Japanese shiba dog",
        "shiba inu outdoor",
        "shiba inu portrait",
    ]:
        try:
            for url, title in wikimedia_search(q, limit=50):
                if url not in all_urls:
                    all_urls[url] = title
            time.sleep(0.8)
        except Exception as e:
            print(f"    '{q}': {e}")
    print(f"  累計 {len(all_urls)}件")

    # 3. Wikimedia カテゴリ
    print("[3/5] Wikimedia カテゴリ...")
    for cat in [
        "Category:Shiba Inu",
        "Category:Shiba inu",
        "Category:Shiba (dog)",
        "Category:Dogs of Japan",
    ]:
        try:
            for url, title in wikimedia_category(cat, limit=50):
                if url not in all_urls:
                    all_urls[url] = title
            time.sleep(0.8)
        except Exception as e:
            print(f"    '{cat}': {e}")
    print(f"  累計 {len(all_urls)}件")

    # 4. English Wikipedia
    print("[4/5] English Wikipedia 記事...")
    wp = wikipedia_images(["Shiba Inu", "Japanese dog breeds", "Shiba (name)"])
    for url, title in wp:
        if url not in all_urls:
            all_urls[url] = title
    print(f"  累計 {len(all_urls)}件")

    # 5. Japanese Wikipedia
    print("[5/5] 日本語 Wikipedia 記事...")
    jawp = jawiki_images(["柴犬", "日本犬"])
    for url, title in jawp:
        if url not in all_urls:
            all_urls[url] = title
    print(f"  累計 {len(all_urls)}件\n")

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
            if downloaded % 10 == 0 or downloaded <= 5:
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
        print("\n目標150枚達成!")
    else:
        print(f"\nあと{150 - total}枚不足。手動で data/raw/shiba/ に追加してください。")


if __name__ == "__main__":
    main()
