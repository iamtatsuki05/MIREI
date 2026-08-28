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
python scripts/constract_llm/train/ft/run_st.py config/constract_llm/train/ft/Sentence-ModernBERT-JP-0.5B/ft.json
```

マルチGPU学習の場合：

```bash
uv run torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node $NUM_GPU \
  scripts/constract_llm/train/ft/run_st.py config/constract_llm/train/ft/Sentence-ModernBERT-JP-0.5B/ft.json
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

詳細なパラメータ情報については、`src/mirei/constract_llm/train/st/data_class/`のデータクラスを参照してください。

### モデルパラメータ

- `model_name_or_path`: 事前学習済みモデルへのパスまたはHugging Faceハブからのモデル識別子
- `loss_cache_mini_batch_size`: 損失キャッシュのミニバッチサイズ
- `loss_scale`: 損失のスケール係数
- `torch_dtype`: モデルの重みのデータ型（float16、float32など）
- `attn_implementation`: 使用するアテンション実装（例：'flash_attention_2'）
- `low_cpu_mem_usage`: モデルのロード時に低CPUメモリ使用量を使用するかどうか

## 設定ファイル

FTの設定ファイルは`config/constract_llm/train/ft/`に格納されています：

- `Sentence-Llama-Bi-JP-0.5B/pt.json` - 事前学習済みLlama-Bi-JP-0.5BをSentence TransformerとしてWSLするための設定
- `Sentence-Llama-Bi-JP-0.5B/ft.json` - Llama-Bi-JP-0.5BをSentence TransformerとしてFTするための設定
- `Sentence-Llama-Bi-EN-0.5B/pt.json` / `Sentence-Llama-Bi-EN-0.5B/ft.json` - Llama-Bi-0.5Bの英語WSL/FT設定
- `Sentence-Llama-Bi-JP-1B/pt-bs8192.json` / `Sentence-Llama-Bi-JP-1B/ft-bs1024.json` - Llama-Bi-1Bの日本語WSL/FT設定
- `Sentence-Llama-Bi-EN-1B/pt-bs8192.json` / `Sentence-Llama-Bi-EN-1B/ft-bs1024.json` - Llama-Bi-1Bの英語WSL/FT設定
- `Sentence-Llama-Bi-JP-3B/pt-bs16384.json` / `Sentence-Llama-Bi-JP-3B/ft-bs2048.json` - Llama-Bi-3Bの日本語WSL/FT設定
- `Sentence-Llama-Bi-EN-3B/pt-bs16384.json` / `Sentence-Llama-Bi-EN-3B/ft-bs2048.json` - Llama-Bi-3Bの英語WSL/FT設定
- `Sentence-ModernBERT-JP-0.5B/pt.json` - 事前学習済みModernBERT-JP-0.5BをSentence TransformerとしてWSLするための設定
- `Sentence-ModernBERT-JP-0.5B/ft.json` - ModernBERT-JP-0.5BをSentence TransformerとしてFTするための設定
- `Sentence-ModernBERT-EN-0.5B/pt.json` / `Sentence-ModernBERT-EN-0.5B/ft.json` - ModernBERT-0.5Bの英語WSL/FT設定
- `Sentence-ModernBERT-JP-1B/pt-bs8192.json` / `Sentence-ModernBERT-JP-1B/ft-bs1024.json` - ModernBERT-1Bの日本語WSL/FT設定
- `Sentence-ModernBERT-EN-1B/pt-bs8192.json` / `Sentence-ModernBERT-EN-1B/ft-bs1024.json` - ModernBERT-1Bの英語WSL/FT設定
- `Sentence-ModernBERT-JP-3B/pt-bs16384.json` / `Sentence-ModernBERT-JP-3B/ft-bs2048.json` - ModernBERT-3Bの日本語WSL/FT設定
- `Sentence-ModernBERT-EN-3B/pt-bs16384.json` / `Sentence-ModernBERT-EN-3B/ft-bs2048.json` - ModernBERT-3Bの英語WSL/FT設定
- `Sentence-Sarashina-Bi-0.5B/pt.json` / `Sentence-Sarashina-Bi-0.5B/ft.json` - Sarashina-Bi-0.5Bの日本語WSL/FT設定
- `Sentence-Sarashina-Bi-EN-0.5B/pt.json` / `Sentence-Sarashina-Bi-EN-0.5B/ft.json` - Sarashina-Bi-0.5Bの英語WSL/FT設定
- `Sentence-Sarashina-Bi-JP-1B/pt-bs8192.json` / `Sentence-Sarashina-Bi-JP-1B/ft-bs1024.json` - Sarashina-Bi-1Bの日本語WSL/FT設定
- `Sentence-Sarashina-Bi-EN-1B/pt-bs8192.json` / `Sentence-Sarashina-Bi-EN-1B/ft-bs1024.json` - Sarashina-Bi-1Bの英語WSL/FT設定
- `Sentence-Sarashina-Bi-JP-3B/pt-bs16384.json` / `Sentence-Sarashina-Bi-JP-3B/ft-bs2048.json` - Sarashina-Bi-3Bの日本語WSL/FT設定
- `Sentence-Sarashina-Bi-EN-3B/pt-bs16384.json` / `Sentence-Sarashina-Bi-EN-3B/ft-bs2048.json` - Sarashina-Bi-3Bの英語WSL/FT設定
