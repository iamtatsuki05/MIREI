# 学習スクリプト

[English](README.md) / 日本語

このディレクトリには、言語モデルの事前学習および Sentence Transformer FineTuneを起動するスクリプトが含まれています。

## ディレクトリ構成

- `pt/` – 事前学習用
  - `run_mlm.py` – Masked Language Modeling（エンコーダモデル向け）。
  - `run_mntp.py` – Masked Next Token Prediction（LoRA 対応の因果言語モデル向け）。
  - `run_clm.py` – Causal Language Modeling（デコーダ型モデル向け）。
- `ft/` – Sentence Transformer FineTune
  - `run_st.py` – Sentence-Transformers Trainer によるコントラスト学習／トリプレット学習。

## 事前学習

### `run_mlm.py`
- Hugging Face Transformers の MLM パイプラインをラップ（BERT、RoBERTa、ModernBERT 等）。
- 行単位／連結テキストの両方に対応。
- DeepSpeed や DDP と組み合わせて使用可能。

### `run_mntp.py`
- 因果モデルに対するマスク付き次トークン予測タスクを実装。
- マスキング戦略、LoRA ランク／ドロップアウト、早期終了などのスイッチを提供。
- 内部的に `AutoModelForCausalLM` を利用。

### `run_clm.py`
- GPT・Llama 等のデコーダ型モデルを自己回帰目的で学習／FineTune。
- `ModelArguments` / `DataTrainingArguments` / `TrainingArguments` を設定ファイルから読み込み。
- ストリーミングデータセット、チェックポイント再開、パープレキシティ評価をサポート。

## FineTune

### `run_st.py`
- Sentence Transformer 向け学習ループを構築し、Triplet評価や IR 評価を実行。
- 複数サブセットの結合、ストリーミング、定数ラベル付与、分散学習に対応。
- アンカー／ポジティブ／ネガティブ列や評価器種別など細かな設定が可能。

## 実行方法

すべてのランチャーは [Google Fire](https://github.com/google/python-fire) を利用した CLI を公開しており、JSON/YAML/TOML の設定ファイルを受け取ります。

```bash
python scripts/constract_llm/train/pt/run_mlm.py config/constract_llm/train/pt/ModernBERT-JP-0.5B-PT-stage1.json
```

マルチ GPU の場合は `uv run torchrun --standalone --nnodes 1 --nproc-per-node $NUM_GPU` をコマンド前に付けて実行してください。

## 設定ファイル

設定テンプレートは `config/constract_llm/train/` 配下に用意されています。

- 事前学習（`pt/`）：
  - `Llama-JP-0.5B-PT-stage1.json`
  - `Llama-JP-0.5B-PT-stage2.json`
  - `ModernBERT-JP-0.5B-PT-stage1.json`
  - `ModernBERT-JP-0.5B-PT-stage2.json`
- ファインチューニング（`ft/`）：
  - `Sentence-Llama-Bi-JP-0.5B-PT.json`
  - `Sentence-Llama-Bi-JP-0.5B.json`
  - `Sentence-ModernBERT-JP-0.5B-PT.json`
  - `Sentence-ModernBERT-JP-0.5B.json`
  - `Sentence-Sarashina-Bi-0.5B-PT.json`
  - `Sentence-Sarashina-Bi-0.5B.json`
- DeepSpeed 設定: `config/constract_llm/train/ds_config/`

テンプレートをコピーしてハイパーパラメータを調整し、対応するランチャーに渡してください。
