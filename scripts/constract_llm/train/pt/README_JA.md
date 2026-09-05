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
python scripts/constract_llm/train/pt/run_mlm.py config/constract_llm/train/pt/ModernBERT-JP-0.5B/stage1.json
```

マルチGPU学習の場合：

```bash
uv run torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node $NUM_GPU \
  scripts/constract_llm/train/pt/run_mlm.py config/constract_llm/train/pt/ModernBERT-JP-0.5B/stage1.json
```

### カスタムパラメータ

詳細なパラメータ情報については、`src/mirei/constract_llm/train/language_model/mlm/data_class/`のデータクラスを参照してください。

## Masked Next Token Prediction (MNTP)

`run_mntp.py`スクリプトは、マスクされたトークンの次のトークンを予測するようにモデルを訓練するMasked Next Token Predictionの目的関数を使用したモデルの事前学習に使用されます。

### 主な機能

- AutoModelForCausalLMアーキテクチャを使用
- 効率的な学習のためのLoRAパラメータ設定が可能

### 使用方法

```bash
python scripts/constract_llm/train/pt/run_mntp.py config/constract_llm/train/pt/Llama-JP-0.5B/stage1.json
```

マルチGPU学習の場合：

```bash
uv run torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node $NUM_GPU \
  scripts/constract_llm/train/pt/run_mntp.py config/constract_llm/train/pt/Llama-JP-0.5B/stage1.json
```

### カスタムパラメータ

- `mask_token_type`: マスキングに使用するトークンのタイプ（'blank'、'eos'、または'mask'）
- `data_collator_type`: 使用するデータコレーターのタイプ（'default'または'all_mask'）
- `lora`: LoRAを使用するかどうか
- `lora_r`: LoRA行列のランク
- `lora_dropout`: LoRA層のドロップアウト確率
- `stop_after_n_steps`: 学習を停止するステップ数

詳細なパラメータ情報については、`src/mirei/constract_llm/train/language_model/mntp/data_class/`のデータクラスを参照してください。

## Sequence Packing（文書境界を守る packing）

`run_mlm.py` と `run_clm.py` は共通の packing オプションを受け付けます（実装は
`src/mirei/constract_llm/train/language_model/packing.py`）。文書境界を越える注意は発生しないため、文書単位で学習した場合と
結果が一致したまま、短い文書が多いデータで GPU 利用率を上げられます。

| パラメータ | 既定値 | 説明 |
|---|---|---|
| `packing` | `false` | 複数の文書を 1 行に詰める（`trl.data_utils.pack_dataset` を使用）。MLM では `line_by_line: true` が必要。 |
| `packing_strategy` | `bfd` | `bfd` は文書を分断しない（best-fit decreasing。行より長い文書は切り詰め）。`wrapped` は `group_texts` と同じ連結・分割（境界情報なし）。 |
| `packing_seq_length` | `max_seq_length` | 1 行のトークン数。 |
| `packing_encoder_mode`（MLM） | `auto` | `unpad`: 行を文書に分け直し、ModernBERT が `cu_seqlens` で再度詰める（`flash_attention_2` で本来の packing）。`mask`: 3D のブロック対角 attention mask と文書ごとの `position_ids` を渡す（BERT 系向け）。`auto` は ModernBERT なら `unpad`、それ以外は `mask`。 |
| `packing_mask_document_starts`（CLM） | `true` | 各文書の先頭トークンを CLM の損失から除外する。 |

decoder には文書ごとに振り直した `position_ids` と FlashAttention の varlen 引数（`cu_seq_lens_q/k`、`max_length_q/k`）を渡し、
`attention_mask` は渡しません。`flash_attention_2` は varlen 引数をそのまま使い、`sdpa` / `eager` は transformers（4.53 以降、
torch 2.6 以降）が `position_ids` の折り返しからブロック対角マスクを組み立てます。後者は KV cache があると無効になるため、
スクリプトは packing 時に `use_cache=false` を設定します（保存される `config.json` にも残るので、生成時に必要なら戻してください）。
古いライブラリでは境界なしで黙って学習せず、起動時にエラーにします。`wrapped` では境界処理は一切行いません。packing 時の
「バッチ」は詰めた行の数なので、マイクロバッチあたりおおよそ `per_device_train_batch_size × packing_seq_length` トークンになり、
`packing_seq_length` は `max_seq_length` 以下、MLM の packing は `pad_to_max_length` と併用不可です。

## 設定ファイル

事前学習の設定ファイルは`config/constract_llm/train/pt/`に格納されています：

- `Llama-JP-0.5B/stage1.json` - Llama-JP-0.5Bのステージ1事前学習の設定
- `Llama-JP-0.5B/stage2.json` - Llama-JP-0.5Bのステージ2事前学習の設定
- `Llama-EN-0.5B/stage1.json` - Llama-0.5Bの英語ステージ1事前学習の設定
- `Llama-EN-0.5B/stage2.json` - Llama-0.5Bの英語ステージ2事前学習の設定
- `Llama-JP-1B/stage1.json`(total batch を 0.5B と統一) / `Llama-JP-1B/stage1-bs8192.json` / `Llama-JP-1B/stage2.json`(total batch を 0.5B と統一) / `Llama-JP-1B/stage2-bs1024.json` - Llama-1Bの日本語中間スケール実験設定
- `Llama-EN-1B/stage1.json`(total batch を 0.5B と統一) / `Llama-EN-1B/stage1-bs8192.json` / `Llama-EN-1B/stage2.json`(total batch を 0.5B と統一) / `Llama-EN-1B/stage2-bs1024.json` - Llama-1Bの英語中間スケール実験設定
- `Llama-JP-3B/stage1.json`(total batch を 0.5B と統一) / `Llama-JP-3B/stage1-bs16384.json` / `Llama-JP-3B/stage2.json`(total batch を 0.5B と統一) / `Llama-JP-3B/stage2-bs2048.json` - Llama-3Bの日本語スケール実験設定
- `Llama-EN-3B/stage1.json`(total batch を 0.5B と統一) / `Llama-EN-3B/stage1-bs16384.json` / `Llama-EN-3B/stage2.json`(total batch を 0.5B と統一) / `Llama-EN-3B/stage2-bs2048.json` - Llama-3Bの英語スケール実験設定
- `ModernBERT-JP-0.5B/stage1.json` - ModernBERT-JP-0.5Bのステージ1事前学習の設定
- `ModernBERT-JP-0.5B/stage2.json` - ModernBERT-JP-0.5Bのステージ2事前学習の設定
- `ModernBERT-EN-0.5B/stage1.json` - ModernBERT-0.5Bの英語ステージ1事前学習の設定
- `ModernBERT-EN-0.5B/stage2.json` - ModernBERT-0.5Bの英語ステージ2事前学習の設定
- `ModernBERT-JP-1B/stage1.json`(total batch を 0.5B と統一) / `ModernBERT-JP-1B/stage1-bs8192.json` / `ModernBERT-JP-1B/stage2.json`(total batch を 0.5B と統一) / `ModernBERT-JP-1B/stage2-bs1024.json` - ModernBERT-1Bの日本語中間スケール実験設定
- `ModernBERT-EN-1B/stage1.json`(total batch を 0.5B と統一) / `ModernBERT-EN-1B/stage1-bs8192.json` / `ModernBERT-EN-1B/stage2.json`(total batch を 0.5B と統一) / `ModernBERT-EN-1B/stage2-bs1024.json` - ModernBERT-1Bの英語中間スケール実験設定
- `ModernBERT-JP-3B/stage1.json`(total batch を 0.5B と統一) / `ModernBERT-JP-3B/stage1-bs16384.json` / `ModernBERT-JP-3B/stage2.json`(total batch を 0.5B と統一) / `ModernBERT-JP-3B/stage2-bs2048.json` - ModernBERT-3Bの日本語スケール実験設定
- `ModernBERT-EN-3B/stage1.json`(total batch を 0.5B と統一) / `ModernBERT-EN-3B/stage1-bs16384.json` / `ModernBERT-EN-3B/stage2.json`(total batch を 0.5B と統一) / `ModernBERT-EN-3B/stage2-bs2048.json` - ModernBERT-3Bの英語スケール実験設定


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
python scripts/constract_llm/train/pt/run_clm.py config/constract_llm/train/pt/YourCLMConfig/stage2.json
```

マルチGPU学習の場合：

```bash
uv run torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node $NUM_GPU \
  scripts/constract_llm/train/pt/run_clm.py config/constract_llm/train/pt/YourCLMConfig/stage2.json
```

### カスタムパラメータ

詳細なパラメータ情報については、`src/mirei/constract_llm/train/language_model/clm/data_class/`のデータクラスを参照してください。
