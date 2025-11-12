# スクリプト一覧

[English](README.md) / 日本語

このフォルダには Text Embedding プロジェクトの自動化エントリーポイントがまとまっています。各スクリプトは `src/mirei` 以下の機能をラップし、データセット処理からモデル・トークナイザーの操作、評価ジョブまでコマンドラインで実行できます。

## ディレクトリ概要

- `main.py` – 「Hello, World!」を表示する最小のデモで、Google Fire ベースの CLI の使い方を示します。
- `constract_llm/` – 言語モデルの構築・学習・パッケージ化・評価を支援するワークフロー用ユーティリティ。
  - `dataset/` – クレンジング、前処理、分割、ハードネガティブマイニングの補助スクリプト。
  - `tokenizer/` – 新規トークナイザーの学習や既存トークナイザーの拡張・マージ用コマンド。
  - `model/` – ベースモデルの初期化やタスク特化バンドルのエクスポート用ツール。
  - `train/` – 言語モデル事前学習（`pt/`）と Sentence Transformer FineTune（`ft/`）の起動スクリプト。
  - `eval/` – 埋め込みベンチマーク、等方性指標、JMTEB ランナー、JGLUE ラッパー等の評価補助。

## 共通の利用パターン

全てのスクリプトは [Google Fire](https://github.com/google/python-fire) を用いたコマンドを公開しています。設定は通常 `config/constract_llm/...` に置いた JSON/YAML/TOML から読み込み、CLI 引数で個別の値を上書きできます。

```bash
python scripts/constract_llm/dataset/cleanse/cleanse.py config/constract_llm/dataset/cleanse/config.json
python scripts/constract_llm/dataset/cleanse/cleanse.py config/constract_llm/dataset/cleanse/config.json --do_deduplicate=False
```

## 設定ファイルの場所

主な設定ディレクトリ：

- `config/constract_llm/dataset/` – クレンジング、前処理、分割ジョブ。
- `config/constract_llm/tokenizer/` – 学習、SentencePieceマージ、トークン追加。
- `config/constract_llm/model/` – モデル初期化とカスタムパッケージ出力。
- `config/constract_llm/train/` – 事前学習（`pt/`）、FineTune（`ft/`）、DeepSpeed 設定。
- `config/constract_llm/eval/` – 等方性評価のサンプルやタスク別設定。

実行前にこれらのテンプレートをコピーし、プロジェクトに合わせて調整してください。
