# 3D Gaussian Splatting × 災害対応 研究サーベイ

## 概要

3D Gaussian Splatting (3DGS) を災害対応・被害評価に応用する研究分野のサーベイ。
この分野は **2024-2025年に急速に立ち上がりつつある新興領域** であり、直接的な論文はまだ少ないが、関連技術の応用可能性は非常に高い。

---

## 1. 直接関連する研究（3DGS × 災害）

### 1.1 建物避難評価
**Reconstruction of Existing Buildings for Evacuation Assessment Under Emergency Situations Using 3D Gaussian Splatting and Machine Learning**
- 著者: Xiaofeng Liao, Bingyang Zhou, Yiqiao Liu, Liwen Zhang, Zhaoyin Zhou, Heap-Yih Chong
- 年: 2025
- 掲載: Reliability Engineering & System Safety, Vol. 267
- 概要: 3DGSによる建物BIM再構築 → 火災・避難シミュレーション → CNN-LSTMによる避難時間予測。リアルタイム緊急時意思決定を支援。
- URL: https://www.sciencedirect.com/science/article/abs/pii/S0951832025010877

### 1.2 山火事SLAM
**WildfireX-SLAM: A Large-Scale Low-Altitude RGB-D Dataset for Wildfire SLAM and Beyond**
- 著者: Zhicong Sun et al.
- 年: 2025
- 概要: Unreal Engine 5で構築した16km²の森林マップ上で5,500枚のRGB-D航空画像データセットを提供。3DGS-SLAM手法のベンチマーク。山火事緊急対応と森林管理がターゲット。
- URL: https://arxiv.org/abs/2510.27133
- プロジェクト: https://zhicongsun.github.io/wildfirexslam

### 1.3 車両損傷評価
**CrashSplat: 2D to 3D Vehicle Damage Segmentation in Gaussian Splatting**
- 著者: Dragos-Andrei Chileban, Andrei-Stefan Bulzan, Cosmin Cernazanu-Glavan
- 年: 2025
- 概要: YOLOベースの2D損傷セグメンテーション → 3DGSへの投影。保険・災害後の車両評価に応用可能。
- URL: https://arxiv.org/abs/2509.23947
- GitHub: https://github.com/DragosChileban/CrashSplat

### 1.4 文化財損傷記録
**3D Visualization of Damaged Statues Using Gaussian Splatting and Web Interface Integration**
- 年: 2025
- 掲載: npj Heritage Science
- 概要: 損傷した文化財の3DGS再構築。セマンティックセグメンテーションによる損傷マスク、PCA分析、対称性評価で修復ガイドを提供。
- URL: https://www.nature.com/articles/s40494-025-02063-5

### 1.5 インフラひび割れ検出
**Adaptive View 3D Gaussian Splatting (AVGS) for Crack Detection**
- 年: 2025
- 概要: SAM（Segment Anything Model）の2Dひび割れセグメンテーション + AVGSによる3Dひび割れ再構築。構造ヘルスモニタリングに応用。
- URL: https://www.sciencedirect.com/science/article/abs/pii/S0888327025014724

### 1.6 ドローン災害復旧ツール
**SkySplat**
- 年: 2024-2025
- 概要: ドローン + Gaussian Splattingによる災害復旧・緊急対応向け自動3Dモデリングツール。2010年ハイチ地震が着想の起点。Blenderアドオンとしてオープンソース化。
- URL: https://devpost.com/software/skysplat | https://skysplat.org/

---

## 2. 関連研究（NeRF / 3D再構築 × 災害）

### 2.1 3Dピクセル単位損傷マッピング
**3D Pixelwise Damage Mapping Using a Deep Attention-Based Modified Nerfacto**
- 著者: Geontae Kim, Youngjin Cha
- 年: 2024
- 掲載: Automation in Construction, Vol. 168
- 概要: NeRFベースの3D再構築 + ひび割れセグメンテーション。橋梁インフラ検査のデジタルツイン。
- URL: https://www.sciencedirect.com/science/article/pii/S0926580524006149

### 2.2 大規模航空シーン再構築
**BirdNeRF: Fast Neural Reconstruction of Large-Scale Scenes from Aerial Imagery**
- 年: 2024-2025
- 掲載: Scientific Reports
- 概要: 鳥瞰ポーズベースの空間分解で大規模航空シーンNeRF再構築。従来の10倍高速。災害対応を主要応用として明示。
- URL: https://arxiv.org/abs/2402.04554

### 2.3 災害後評価3Dベンチマーク
**3DAeroRelief: The First 3D Benchmark UAV Dataset for Post-Disaster Assessment**
- 年: 2025
- 概要: ハリケーン被害地域のUAV収集3D点群データセット。災害後評価に特化した初の3Dベンチマーク。
- URL: https://arxiv.org/abs/2509.11097

### 2.4 ドローン+地上のガウシアンスプラッティング
**DRAGON: Drone and Ground Gaussian Splatting for 3D Building Reconstruction**
- 著者: Yujin Ham et al.
- 年: 2024
- 掲載: IEEE ICCP
- 概要: ドローン（航空）+ 地上（携帯電話）画像の統合による建物3D再構築。災害後の建物評価に直接応用可能。
- URL: https://arxiv.org/abs/2407.01761

### 2.5 野外ドローン画像からの3DGS
**DroneSplat: 3D Gaussian Splatting for Robust 3D Reconstruction from In-the-Wild Drone Imagery**
- 年: 2025
- 掲載: CVPR 2025
- 概要: 動的障害物の除去、マルチビューステレオ予測を統合したドローン画像からの頑健な3DGS。
- URL: https://arxiv.org/abs/2503.16964
- GitHub: https://github.com/BITyia/DroneSplat

---

## 3. 産業動向

### DJI Terra V5.0 / FlightHub 2 — Gaussian Splatting統合
- DJIがドローン3DモデリングプラットフォームにGaussian Splattingを統合（2024-2025）
- 構造物損傷評価、インフラ検査、緊急対応でのフォトリアリスティック3D再構築が可能に
- URL: https://drone-parts-center.com/en/blog/dji-terra-v5-0-the-gaussian-splatting-revolution-transforms-drone-3d-modeling/

### 大規模都市シーン再構築（ICCV 2025）
**Robust and Efficient 3D Gaussian Splatting for Urban Scene Reconstruction**
- 最大5,000万ガウシアン楕円体による都市再構築。リソース制約環境（災害現場など）でのデプロイを想定。

### 建物再構築
**GS4Buildings: Prior-Guided Gaussian Splatting for Building Reconstruction**
- LoD2セマンティックモデルを事前知識として建物3DGS再構築。災害後の建物被害記録に直接応用可能。
- GitHub: https://github.com/zqlin0521/GS4Buildings

---

## 4. 災害関連データセット

| データセット | 年 | 内容 | モダリティ | URL |
|---|---|---|---|---|
| 3DAeroRelief | 2025 | ハリケーン被害3D点群 | UAV | arXiv:2509.11097 |
| WildfireX-SLAM | 2025 | 山火事森林16km² | RGB-D航空 | arXiv:2510.27133 |
| DisasterM3 | 2025 | グローバル災害VLデータセット | 衛星+SAR | arXiv:2505.21089 |
| BRIGHT | 2025 | 建物被害評価 | 光学+SAR衛星 | arXiv:2501.06019 |

---

## 5. 研究ギャップ（未開拓の研究機会）

以下のテーマはまだ論文が存在せず、重要な研究機会となる：

1. **リアルタイム地震後都市被害評価 × 3DGS** — 地震直後のドローン映像から3DGSで迅速に被害マップを生成
2. **洪水範囲マッピング × 3DGS** — 時系列3DGSによる浸水域の可視化と変化検出
3. **マルチテンポラル3DGS変化検出** — 災害前後の3DGSモデル比較による被害自動検出
4. **リアルタイムドローンSLAM × 3DGS × 災害対応** — オンボードでの3DGS処理による即座の状況把握
5. **大規模災害デジタルツイン** — 都市規模の3DGSモデルと被害シミュレーションの統合
6. **マルチモーダル3DGS（RGB + 熱赤外 + LiDAR）** — 火災・生存者捜索における複合センシング

---

## 6. 提案する研究方向性

### 方向性A: 地震後迅速被害評価パイプライン
ドローン映像 → リアルタイム3DGS再構築 → 被害セグメンテーション → 被害度マップ生成
- 技術基盤: DroneSplat + AVGS + 被害分類モデル
- 差別化: エンドツーエンドのリアルタイムパイプライン（既存研究はオフライン処理）

### 方向性B: マルチテンポラル3DGS変化検出
災害前の3DGSモデル ↔ 災害後の3DGSモデル → 自動差分検出 → 被害度分類
- 技術基盤: 3DGS + 点群比較 + 変化検出アルゴリズム
- 差別化: 3DGSの高速再構築能力を活かした時系列比較（NeRFでは遅すぎる）

### 方向性C: 3DGS災害デジタルツイン
都市規模3DGS + 物理シミュレーション（地震・洪水・火災） → 被害予測・避難計画
- 技術基盤: Liao et al. (2025) の拡張 + 大規模都市3DGS
- 差別化: 建物単体→都市規模へのスケールアップ

---

## ファイル構成

```
research/3dgs_disaster/
├── README.md              # 本ファイル（サーベイまとめ）
├── papers/                # 論文PDF・メモ
├── proposals/             # 研究提案書
└── experiments/           # 実験コード・結果
```

---

*最終更新: 2026-03-27*
