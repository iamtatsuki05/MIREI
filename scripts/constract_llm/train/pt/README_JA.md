# 事前学習スクリプト

[English](README.md) / 日本語

このディレクトリには、言語モデルの事前学習に関連するスクリプトが含まれています。

## 利用可能なスクリプト

- `run_mlm.py` - Masked Language Modeling（マスク言語モデリング）事前学習用スクリプト
- `run_mntp.py` - Masked Next Token Prediction（マスク次トークン予測）事前学習用スクリプト
- `run_clm.py` - Causal Language Modeling（自己回帰型言語モデリング）事前学習・ファインチューニング用スクリプト（GPT, GPT-2, Llama等）

## Masked Language Modeling (MLM)

`run_mlm.py`スクリプトは、入力のランダムなトークンをマスクし、モデルが元のトークンを予測するように訓練するMasked Language Modelingの目的関数を使用したモデルの事前学習に使用されます。

### 主な機能

- 様々なモデルアーキテクチャ（BERT、RoBERTaなど）をサポート
- マスキング確率の設定が可能
- 行単位とテキスト連結の両方の処理をサポート
- 精度メトリクスによる評価に対応

### 使用方法

```bash
python scripts/constract_llm/train/pt/run_mlm.py config/constract_llm/train/pt/ModernBERT-JP-0.5B-PT-stage1.json
```

マルチGPU学習の場合：

```bash
uv run torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node $NUM_GPU \
  scripts/constract_llm/train/pt/run_mlm.py config/constract_llm/train/pt/ModernBERT-JP-0.5B-PT-stage1.json
```

### カスタムパラメータ

詳細なパラメータ情報については、`src/nlp/constract_llm/train/language_model/mlm/data_class/`のデータクラスを参照してください。

## Masked Next Token Prediction (MNTP)

`run_mntp.py`スクリプトは、マスクされたトークンの次のトークンを予測するようにモデルを訓練するMasked Next Token Predictionの目的関数を使用したモデルの事前学習に使用されます。

### 主な機能

- AutoModelForCausalLMアーキテクチャを使用
- 効率的な学習のためのLoRAパラメータ設定が可能

### 使用方法

```bash
python scripts/constract_llm/train/pt/run_mntp.py config/constract_llm/train/pt/Llama-JP-0.5B-PT-stage1.json
```

マルチGPU学習の場合：

```bash
uv run torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node $NUM_GPU \
  scripts/constract_llm/train/pt/run_mntp.py config/constract_llm/train/pt/Llama-JP-0.5B-PT-stage1.json
```

### カスタムパラメータ

- `mask_token_type`: マスキングに使用するトークンのタイプ（'blank'、'eos'、または'mask'）
- `data_collator_type`: 使用するデータコレーターのタイプ（'default'または'all_mask'）
- `lora`: LoRAを使用するかどうか
- `lora_r`: LoRA行列のランク
- `lora_dropout`: LoRA層のドロップアウト確率
- `stop_after_n_steps`: 学習を停止するステップ数

詳細なパラメータ情報については、`src/nlp/constract_llm/train/language_model/mntp/data_class/`のデータクラスを参照してください。

## 設定ファイル

事前学習の設定ファイルは`config/constract_llm/train/pt/`に格納されています：

- `Llama-JP-0.5B-PT-stage1.json` - Llama-JP-0.5Bのステージ1事前学習の設定
- `Llama-JP-0.5B-PT-stage2.json` - Llama-JP-0.5Bのステージ2事前学習の設定
- `ModernBERT-JP-0.5B-PT-stage1.json` - ModernBERT-JP-0.5Bのステージ1事前学習の設定
- `ModernBERT-JP-0.5B-PT-stage2.json` - ModernBERT-JP-0.5Bのステージ2事前学習の設定


## Causal Language Modeling (CLM)

`run_clm.py`スクリプトは、Causal Language Modeling（自己回帰型言語モデリング）の目的関数を用いたモデルの事前学習・ファインチューニングに使用します。
これは、GPT、GPT-2、Llamaなどのデコーダ型アーキテクチャに適しています。モデルは系列内の次のトークンを逐次予測するように訓練されます。

### 主な機能

- HuggingFaceの`AutoModelForCausalLM`および互換アーキテクチャをサポート
- 柔軟なデータセット読み込み（ローカルファイル・HuggingFace Datasets両対応）
- JSON設定ファイル（ModelArguments, DataTrainingArguments, TrainingArguments）による柔軟な設定
- `torchrun`による分散・マルチGPU学習に対応
- パープレキシティ・精度による評価
- チェックポイントからの再開学習

### 使用方法

```bash
python scripts/constract_llm/train/pt/run_clm.py config/constract_llm/train/pt/YourCLMConfig.json
```

マルチGPU学習の場合：

```bash
uv run torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node $NUM_GPU \
  scripts/constract_llm/train/pt/run_clm.py config/constract_llm/train/pt/YourCLMConfig.json
```

### カスタムパラメータ

詳細なパラメータ情報については、`src/nlp/constract_llm/train/language_model/clm/data_class/`のデータクラスを参照してください。
