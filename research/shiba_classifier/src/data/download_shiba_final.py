"""
柴犬の実写画像を150枚以上収集 (レートリミット対応版)

429エラー対策: DL間隔2秒、429時は指数バックオフで再試行
"""

import json
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
    "calligraphy", "screenshot", "svg",
]

API_UA = {"User-Agent": "ShibaClassifierBot/1.0 (academic research)"}
BROWSER_UA = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def is_rejected(url: str, title: str = "") -> bool:
    text = (url + " " + title).lower()
    return any(kw in text for kw in REJECT_KEYWORDS)


def api_get(url: str) -> dict:
    req = urllib.request.Request(url, headers=API_UA)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def download_with_retry(url: str, dest: Path, max_retries: int = 3) -> bool:
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=BROWSER_UA)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if len(data) < 5_000:
                return False
            if not (data[:2] == b'\xff\xd8' or data[:4] == b'\x89PNG'):
                return False
            dest.write_bytes(data)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 5 * (2 ** attempt)
                print(f"    429 rate limit, waiting {wait}s...")
                time.sleep(wait)
                continue
            return False
        except Exception:
            return False
    return False


def content_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# ── Sources ──

def fetch_dog_ceo() -> List[str]:
    req = urllib.request.Request("https://dog.ceo/api/breed/shiba/images", headers=BROWSER_UA)
    data = json.loads(urllib.request.urlopen(req, timeout=30).read())
    return data.get("message", []) if data.get("status") == "success" else []


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


def wikimedia_category_recursive(category: str, limit: int = 200) -> List[Tuple[str, str]]:
    """カテゴリとサブカテゴリから画像を収集。"""
    results = []
    visited = set()

    def _crawl(cat, depth=0):
        if cat in visited or depth > 2:
            return
        visited.add(cat)
        params = (
            f"action=query&generator=categorymembers"
            f"&gcmtitle={urllib.request.quote(cat)}"
            f"&gcmtype=file|subcat&gcmlimit=50"
            f"&prop=imageinfo&iiprop=url|mime|size"
            f"&format=json"
        )
        try:
            data = api_get(f"https://commons.wikimedia.org/w/api.php?{params}")
        except Exception:
            return

        for page in data.get("query", {}).get("pages", {}).values():
            title = page.get("title", "")
            if title.startswith("Category:"):
                _crawl(title, depth + 1)
                continue
            for ii in page.get("imageinfo", []):
                if ii.get("mime") in ("image/jpeg", "image/png"):
                    results.append((ii["url"], title))
        time.sleep(0.5)

    _crawl(category)
    return results


def wikipedia_images(base_url: str, articles: List[str]) -> List[Tuple[str, str]]:
    results = []
    for article in articles:
        try:
            params = (
                f"action=query&titles={urllib.request.quote(article)}"
                f"&prop=images&imlimit=50&format=json"
            )
            data = api_get(f"{base_url}?{params}")
            for page in data.get("query", {}).get("pages", {}).values():
                for img in page.get("images", []):
                    t = img.get("title", "")
                    if not t.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    info_params = (
                        f"action=query&titles={urllib.request.quote(t)}"
                        f"&prop=imageinfo&iiprop=url|mime&format=json"
                    )
                    info = api_get(f"{base_url}?{info_params}")
                    for p in info.get("query", {}).get("pages", {}).values():
                        for ii in p.get("imageinfo", []):
                            if ii.get("mime") in ("image/jpeg", "image/png"):
                                results.append((ii["url"], t))
            time.sleep(0.5)
        except Exception as e:
            print(f"    '{article}': {e}")
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

    all_urls: Dict[str, str] = {}

    # 1. Dog CEO
    print("[1/5] Dog CEO API...")
    for u in fetch_dog_ceo():
        all_urls[u] = ""
    print(f"  {len(all_urls)}件")

    # 2. Wikimedia 検索 (多数のクエリ)
    print("[2/5] Wikimedia Commons 検索...")
    queries = [
        "shiba inu dog photo", "shiba inu puppy", "shiba inu",
        "shiba ken", "柴犬", "Japanese shiba",
        "shiba inu outdoor", "shiba inu portrait",
        "shiba inu walk", "shiba inu park",
        "red shiba inu", "black shiba inu",
        "shiba inu snow", "shiba inu garden",
    ]
    for q in queries:
        try:
            for url, title in wikimedia_search(q, limit=50):
                if url not in all_urls:
                    all_urls[url] = title
            time.sleep(0.8)
        except Exception:
            pass
    print(f"  累計 {len(all_urls)}件")

    # 3. Wikimedia カテゴリ (サブカテゴリも探索)
    print("[3/5] Wikimedia カテゴリ (再帰)...")
    for cat in ["Category:Shiba Inu", "Category:Shiba inu", "Category:Dogs of Japan"]:
        for url, title in wikimedia_category_recursive(cat):
            if url not in all_urls:
                all_urls[url] = title
    print(f"  累計 {len(all_urls)}件")

    # 4. English Wikipedia
    print("[4/5] English Wikipedia...")
    for url, title in wikipedia_images(
        "https://en.wikipedia.org/w/api.php",
        ["Shiba Inu", "Japanese dog breeds"]
    ):
        if url not in all_urls:
            all_urls[url] = title
    print(f"  累計 {len(all_urls)}件")

    # 5. Japanese Wikipedia
    print("[5/5] 日本語Wikipedia...")
    for url, title in wikipedia_images(
        "https://ja.wikipedia.org/w/api.php",
        ["柴犬", "日本犬"]
    ):
        if url not in all_urls:
            all_urls[url] = title
    print(f"  累計 {len(all_urls)}件\n")

    # フィルタ
    filtered = [(u, t) for u, t in all_urls.items() if not is_rejected(u, t)]
    rejected = len(all_urls) - len(filtered)
    print(f"フィルタ: {len(all_urls)} → {len(filtered)}件 (除外{rejected}件)\n")
    print("ダウンロード開始 (間隔2秒, 429時はバックオフ)...\n")

    idx = len(existing_files)
    downloaded = 0
    skipped_dup = 0
    failed = 0

    for url, title in filtered:
        idx += 1
        ext = ".png" if ".png" in url.lower().split("?")[0] else ".jpg"
        filename = f"shiba_{idx:04d}{ext}"
        dest = RAW_DIR / filename

        ok = download_with_retry(url, dest)
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

        time.sleep(2)  # レートリミット対策

    total = len(list(RAW_DIR.glob("*.*")))
    print(f"\n===== 結果 =====")
    print(f"新規DL: {downloaded}枚")
    print(f"重複スキップ: {skipped_dup}枚")
    print(f"失敗: {failed}枚")
    print(f"data/raw/shiba/ 合計: {total}枚")

    if total >= 150:
        print("\n目標150枚達成!")
    else:
        print(f"\nあと{150 - total}枚不足。")


if __name__ == "__main__":
    main()
