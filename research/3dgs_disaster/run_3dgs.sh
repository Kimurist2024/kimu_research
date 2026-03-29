#!/bin/bash
# 3D Gaussian Splatting パイプライン
# 入力: IMG_6356.MOV → 出力: splat.ply
# 実行環境: A100 MIG 3g.40gb (予約1204)

set -e

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
FFMPEG=/home/st6324034/.local/lib/python3.8/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2
OPENSPLAT=/home/st6324034/kimu_research/research/3dgs_disaster/OpenSplat/build/opensplat
MIG_UUID="MIG-85df71d3-d400-56af-881f-6fd230816891"

# === Step 1: 動画からフレーム抽出 (1fps) ===
echo "=== Step 1: Frame extraction ==="
mkdir -p "$BASE_DIR/input"
$FFMPEG -i "$BASE_DIR/IMG_6356.MOV" -vf "fps=1" -q:v 2 "$BASE_DIR/input/frame_%04d.jpg"

# === Step 2: COLMAP (SfM) でカメラポーズ推定 ===
echo "=== Step 2: COLMAP SfM ==="
python3 << 'PYEOF'
import pycolmap
import os

base = os.environ.get("BASE_DIR", ".")
image_dir = os.path.join(base, "input")
db_path = os.path.join(base, "colmap.db")
sparse_dir = os.path.join(base, "sparse")
os.makedirs(sparse_dir, exist_ok=True)

if os.path.exists(db_path):
    os.remove(db_path)

pycolmap.extract_features(
    database_path=db_path,
    image_path=image_dir,
    camera_mode=pycolmap.CameraMode.AUTO,
    sift_options=pycolmap.SiftExtractionOptions(max_num_features=8192),
)

pycolmap.match_exhaustive(database_path=db_path)

maps = pycolmap.incremental_mapping(
    database_path=db_path,
    image_path=image_dir,
    output_path=sparse_dir,
)
print(f"Reconstruction done. {len(maps)} model(s)")
for idx, rec in maps.items():
    print(f"  Model {idx}: {rec.summary()}")
PYEOF

# === Step 3: COLMAP プロジェクト構造を作成 ===
echo "=== Step 3: Create COLMAP project structure ==="
mkdir -p "$BASE_DIR/colmap_project/sparse/0"
cp "$BASE_DIR/sparse/0/"* "$BASE_DIR/colmap_project/sparse/0/"
ln -sf "$BASE_DIR/input" "$BASE_DIR/colmap_project/images"

# === Step 4: OpenSplat で 3DGS 学習 ===
echo "=== Step 4: OpenSplat training ==="
export LD_LIBRARY_PATH=/home/st6324034/.local/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=$MIG_UUID

$OPENSPLAT "$BASE_DIR/colmap_project" \
    -n 2000 \
    -o "$BASE_DIR/splat.ply"

echo "=== Done! Output: $BASE_DIR/splat.ply ==="

# === 可視化: SuperSplat ===
# cd supersplat && npm install && npm run build && npx serve dist -C -l 3000
# ブラウザで http://localhost:3000 を開き、splat.ply をドラッグ&ドロップ
