# FTスクリプト

[English](README.md) / 日本語

このディレクトリには、文埋め込みモデルのFTに関連するスクリプトが含まれています。

## 利用可能なスクリプト

- `run_st.py` - Sentence Transformer用スクリプト

## Sentence Transformer

`run_st.py`スクリプトは、文のエンベディングを生成するように設計されたSentence TransformerモデルのFTに使用されます。

### 主な機能

- トリプレット損失（CachedMultipleNegativesRankingLoss）による学習をサポート
- データセットの読み込みと処理の設定が可能
- トリプレットでのeval

### 使用方法

```bash
python scripts/constract_llm/train/ft/run_st.py config/constract_llm/train/ft/Sentence-ModernBERT-JP-0.5B.json
```

マルチGPU学習の場合：

```bash
uv run torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node $NUM_GPU \
  scripts/constract_llm/train/ft/run_st.py config/constract_llm/train/ft/Sentence-ModernBERT-JP-0.5B.json
```

### カスタムパラメータ

- `anchor_column_name`: アンカー文のカラム名
- `positive_column_name`: ポジティブ文のカラム名
- `negative_column_name`: ネガティブ文のカラム名
- `label_column_name`: ラベルのカラム名（group_by_labelバッチサンプラーで使用）
- `evaluator_type`: 使用する評価器のタイプ（例：'triplet'）
- `max_subset_samples`: サブセットごとの最大サンプル数
- `streaming`: ストリーミングデータセットを使用するかどうか
- `use_all_subset`: 利用可能なすべてのサブセットを使用するかどうか
- `use_subsets`: 使用する特定のサブセットのリスト
- `max_seq_length`: トークン化の最大シーケンス長

詳細なパラメータ情報については、`src/nlp/constract_llm/train/st/data_class/`のデータクラスを参照してください。

### モデルパラメータ

- `model_name_or_path`: 事前学習済みモデルへのパスまたはHugging Faceハブからのモデル識別子
- `loss_cache_mini_batch_size`: 損失キャッシュのミニバッチサイズ
- `loss_scale`: 損失のスケール係数
- `torch_dtype`: モデルの重みのデータ型（float16、float32など）
- `attn_implementation`: 使用するアテンション実装（例：'flash_attention_2'）
- `low_cpu_mem_usage`: モデルのロード時に低CPUメモリ使用量を使用するかどうか

## 設定ファイル

FTの設定ファイルは`config/constract_llm/train/ft/`に格納されています：

- `Sentence-Llama-Bi-JP-0.5B-PT.json` - 事前学習済みLlama-Bi-JP-0.5BをSentence TransformerとしてWSLするための設定
- `Sentence-Llama-Bi-JP-0.5B.json` - Llama-Bi-JP-0.5BをSentence TransformerとしてFTするための設定
- `Sentence-ModernBERT-JP-0.5B-PT.json` - 事前学習済みModernBERT-JP-0.5BをSentence TransformerとしてWSLするための設定
- `Sentence-ModernBERT-JP-0.5B.json` - ModernBERT-JP-0.5BをSentence TransformerとしてFTするための設定
