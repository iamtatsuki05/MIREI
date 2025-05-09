# 学習スクリプト

[English](README.md) / 日本語

このディレクトリには、言語モデルの学習に関連するスクリプトが含まれています。

## ディレクトリ構成

- `pt/`: 言語モデルの事前学習用スクリプト
  - `run_mlm.py` - Masked Language Modeling（マスク言語モデリング）事前学習用スクリプト
  - `run_mntp.py` - Masked Next Token Prediction（マスク次トークン予測）事前学習用スクリプト
- `ft/`: setence tranformer用のスクリプト
  - `run_st.py` - Sentence TransformerFT/WSL用スクリプト

## 事前学習スクリプト

### Masked Language Modeling (MLM)

- `run_mlm.py` - Masked Language Modelingの目的関数を使用したモデルの事前学習用スクリプト。
  - 様々なモデルアーキテクチャ（BERT、RoBERTaなど）をサポート
  - マスキング確率の設定が可能
  - 行単位とテキスト連結の両方の処理をサポート
  - 精度メトリクスによる評価
  - 詳細なパラメータ情報については、[Hugging Face MLMドキュメント](https://huggingface.co/docs/transformers/tasks/masked_language_modeling)を参照

### Masked Next Token Prediction (MNTP)

- `run_mntp.py` - Masked Next Token Predictionの目的関数を使用したモデルの事前学習用スクリプト。
  - AutoModelForCausalLMアーキテクチャを使用
  - 効率的な学習のためのLoRAパラメータ設定が可能
  - 詳細なパラメータ情報については、[Hugging Face Causal LMドキュメント](https://huggingface.co/docs/transformers/tasks/language_modeling)を参照

## FTスクリプト

### Sentence Transformer

- `run_st.py` - Sentence TransformerモデルのFT用スクリプト。
  - トリプレット損失による学習をサポート
  - データセットの読み込みと処理の設定が可能
  - トリプレット評価器による評価
  - カスタムパラメータ：
    - `anchor_column_name`: アンカー文のカラム名
    - `positive_column_name`: ポジティブ文のカラム名
    - `negative_column_name`: ネガティブ文のカラム名
    - `evaluator_type`: 使用する評価器のタイプ（例：'triplet'）
    - `max_subset_samples`: サブセットごとの最大サンプル数
    - `streaming`: ストリーミングデータセットを使用するかどうか

## 使用方法

各スクリプトは[Google Fire](https://github.com/google/python-fire)を使用してCLIインターフェースを提供しています。基本的な使用方法は以下の通りです：

```bash
python scripts/constract_llm/train/pt/run_mlm.py config/constract_llm/train/pt/ModernBERT-JP-1.4B-PT-stage1.json
```

### マルチGPU学習

すべての学習スクリプトはPyTorch Distributed Data Parallel（DDP）を使用したマルチGPU学習をサポートしています。複数のGPUで学習を実行するには、以下のコマンド形式を使用します：

```bash
uv run torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node $NUM_GPU \
  scripts/constract_llm/train/pt/run_mlm.py config/constract_llm/train/pt/ModernBERT-JP-1.4B-PT-stage1.json
```

ここで、`$NUM_GPU`は学習に使用したいGPUの数です。

## 設定ファイル

各スクリプトの設定ファイルは、`config/constract_llm/train/`以下の対応するディレクトリに格納されています：

- 事前学習の設定: `config/constract_llm/train/pt/`
  - `Llama-Bi-JP-1.4B-PT-stage1.json`
  - `Llama-Bi-JP-1.4B-PT-stage2.json`
  - `ModernBERT-JP-1.4B-PT-stage1.json`
  - `ModernBERT-JP-1.4B-PT-stage2.json`
- FTの設定: `config/constract_llm/train/ft/`
  - `Sentence-Llama-Bi-JP-1.4B-PT.json`
  - `Sentence-Llama-Bi-JP-1.4B.json`
  - `Sentence-ModernBERT-JP-1.4B-PT.json`
  - `Sentence-ModernBERT-JP-1.4B.json`

各設定ファイルには、対応するスクリプトで使用される全てのパラメータが含まれています。
