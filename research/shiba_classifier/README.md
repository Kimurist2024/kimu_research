shiba_classifier
==============================

柴犬画像分類器 - 柴犬か否かを判別するシステム (Transfer Learning with MobileNetV2)

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- `make data` や `make train` などのコマンド定義
    ├── README.md          <- このファイル
    ├── data
    │   ├── external       <- サードパーティのデータ
    │   ├── interim        <- 変換途中の中間データ
    │   ├── processed      <- モデル学習用の最終データセット
    │   └── raw            <- 元の生データ（柴犬画像・非柴犬画像）
    │
    ├── docs               <- ドキュメント
    │
    ├── models             <- 学習済みモデル、予測結果、モデルサマリ
    │
    ├── notebooks          <- Jupyter Notebook
    │                         命名規則: `番号-作成者イニシャル-説明`
    │                         例: `1.0-rk-initial-data-exploration`
    │
    ├── references         <- データ辞書、マニュアル、参考資料
    │
    ├── reports            <- HTML, PDF, LaTeX などの生成レポート
    │   └── figures        <- レポート用の図表
    │
    ├── requirements.txt   <- 環境再現用の依存パッケージ一覧
    │                         `pip freeze > requirements.txt` で生成
    │
    ├── setup.py           <- `pip install -e .` でsrcをインポート可能にする
    ├── src                <- プロジェクトのソースコード
    │   ├── __init__.py    <- src を Python モジュールとして認識させる
    │   │
    │   ├── data           <- データのダウンロード・生成スクリプト
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- 生データを特徴量に変換するスクリプト
    │   │   └── build_features.py
    │   │
    │   ├── models         <- モデルの学習・予測スクリプト
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- 可視化スクリプト
    │       └── visualize.py
    │
    └── tox.ini            <- tox 設定ファイル

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
