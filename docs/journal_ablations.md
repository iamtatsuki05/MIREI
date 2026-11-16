# ジャーナル版のアブレーション実験

issue #55 の「必須」の実験に使う config の一覧と実行順です。条件付きの実験の config は、必須の結果を見てから追加します。

## 共通の前提

- アブレーションは EN を主にし、JP は 1B で傾向を確認する程度にする。
- 文埋め込みの学習は既存と同じ 2 段階 (`pt.json` で弱教師あり、`ft.json` で教師あり) で行う。各 config は既存の `Sentence-*` の config を元にしている。変えたのは、モデル、出力先、run 名と、pooling・tokenizer の指定だけ。
- `model_name_or_path` の Hugging Face リポジトリのうち、次のものはまだ存在しない。実行前に作る必要がある。
  - `*-PT-stage1-ckpt<step>`: stage1 の中間 checkpoint (研究室の archive に tar.gz で保存されている) を展開して push する
  - `Llama-Bi-*` / `*-mntp-init`: `save_custom_model.py` で変換して push する
  - `Llama-Bi-EN-*-PT-stage2-mntp`: `stage2-mntp.json` の学習結果を push する
  - `Sentence-*-PT`: 同じディレクトリの `pt.json` の学習結果を push する
- 双方向化は flash_attention_2 の経路でだけ効く (`modeling_bidirectional_llama.py` の `_update_causal_mask` の上書きは、transformers 4.56 の sdpa / eager の経路では使われない)。Llama-Bi 系の config は flash_attention_2 のまま変えない。

## 1. pooling の比較

既存のモデルは、すべて mean pooling で学習している。

| config (`config/constract_llm/train/ft/`) | 出発点 | pooling | tokenizer の変更 |
|---|---|---|---|
| `Sentence-ModernBERT-EN-0.5B-cls` | `ModernBERT-EN-0.5B-PT-stage2` | cls | BOS を付ける |
| `Sentence-Llama-EN-0.5B-causal-lasttoken` | `Llama-EN-0.5B-PT-stage2` (causal のまま) | lasttoken | EOS を付ける |
| `Sentence-Llama-EN-0.5B-causal-mean` | `Llama-EN-0.5B-PT-stage2` (causal のまま) | mean | なし |
| `Sentence-Llama-EN-1B-causal-lasttoken` | `Llama-EN-1B-PT-stage2` | lasttoken | EOS を付ける |
| `Sentence-Llama-EN-3B-causal-lasttoken` | `Llama-EN-3B-PT-stage2` | lasttoken | EOS を付ける |
| `Sentence-Llama-JP-1B-causal-lasttoken` | `Llama-JP-1B-PT-stage2` | lasttoken | EOS を付ける |

- `pooling_mode` / `add_bos_token` / `add_eos_token` は `pt.json` だけに書く。`ft.json` は `pt.json` の学習結果 (SentenceTransformer として保存済み) を読むので、pooling と tokenizer の設定はそのまま引き継がれる。
- 既存の tokenizer は、BOS も EOS も付けない設定になっている。cls と lasttoken を意味のある位置で取るため、BOS / EOS を付ける。
- tokenizer のクラスによっては `add_bos_token` / `add_eos_token` が黙って無視されるので、`run_st.py` は学習前に 1 文を tokenize して、BOS / EOS が実際に付いているかを確認し、付いていなければ止まる。

## 2. 事前学習量の比較

stage1 の約 25% / 約 50% / 100% の時点の checkpoint から、stage2 を通さずに文埋め込みを学習する。step は保存間隔 (100 step) に丸めている。

- 100% の点も stage2 を通さない (`*-PT-stage1` の Hugging Face リポジトリから学習する)。既存の本番モデル (stage2 あり) は、stage2 の効果を見る参考の行として別に並べる。
- 0.5B で以前に行った比較 (`/project/nlp-mire2/tatsuki-o/e3-scaling/`) は、25% / 50% が stage1 の checkpoint、100% が既存の本番モデル (stage2 あり、別の学習 run) で、100% の点だけ条件が違っていた。今回はこれを揃える。

| モデル | 100% とした step | 25% | 50% |
|---|---|---|---|
| Llama-EN-1B / ModernBERT-EN-1B | 4200 | 1100 | 2100 |
| Llama-JP-1B | 4200 | 1100 | 2100 |
| ModernBERT-JP-1B | 4800 | 1200 | 2400 |
| Llama-EN-3B | 13700 | 3400 | 6900 |
| ModernBERT-EN-3B | 13893 | 3500 | 6900 |

- Llama は、`config/constract_llm/model/save_custom_model/llama_bi_*_pt_stage1{_ckpt<step>,}.json` で双方向モデルに変換してから学習する。
- 文埋め込みの config は、25% / 50% が `Sentence-{Llama-Bi,ModernBERT}-*-PT-stage1-ckpt<step>`、100% が `Sentence-{Llama-Bi,ModernBERT}-*-PT-stage1`。
- `*-PT-stage1` のリポジトリが表の「100% とした step」の checkpoint と同じかは、まだ確認していない。

## 3. MNTP を stage2 の学習目的にする

既存の stage2 (データ、系列長 8192、学習率) を変えずに、学習目的だけを CLM から MNTP にする。`mlm_probability` は encoder の stage2 と同じ 0.3。

| 手順 | 0.5B | 1B |
|---|---|---|
| 1. MNTP 用に変換 | `model/save_custom_model/llama_bi_en_05b_pt_stage1_mntp_init.json` | `model/save_custom_model/llama_bi_en_1b_pt_stage1_mntp_init.json` |
| 2. stage2 (MNTP) | `train/pt/Llama-EN-0.5B/stage2-mntp.json` | `train/pt/Llama-EN-1B/stage2-mntp.json` |
| 3. 文埋め込み | `train/ft/Sentence-Llama-Bi-EN-0.5B-PT-stage2-mntp` | `train/ft/Sentence-Llama-Bi-EN-1B-PT-stage2-mntp` |

- mask token は `run_mntp.py` の既定 (`mask_token_type: blank`、`_`) のまま。`_` は既存の tokenizer で 1 トークン (id 290) になることを確認済み。
- 手順 2 の前に、`run_mntp.py` が系列長 8192 と、flash-attention での双方向化で動くことを短い run で確認する。
- 手順 3 の前に、手順 2 の出力の `config.json` の `auto_map.AutoModel` が、非公開の `*-mntp-init` リポジトリを参照していないかを確認する。参照していれば、そのリポジトリへのアクセス権が要る。

## 実行例

```bash
uv run python scripts/constract_llm/model/save_custom_model.py config/constract_llm/model/save_custom_model/llama_bi_en_1b_pt_stage1_ckpt1100.json
uv run torchrun --standalone --nnodes 1 --nproc-per-node $NUM_GPU scripts/constract_llm/train/pt/run_mntp.py config/constract_llm/train/pt/Llama-EN-1B/stage2-mntp.json
uv run torchrun --standalone --nnodes 1 --nproc-per-node $NUM_GPU scripts/constract_llm/train/ft/run_st.py config/constract_llm/train/ft/Sentence-Llama-EN-1B-causal-lasttoken/pt.json
```
