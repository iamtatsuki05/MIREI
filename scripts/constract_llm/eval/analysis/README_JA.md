# 埋め込み幾何・attention 解析スクリプト(学習なしの追加分析)

MIREI 論文(encoder vs decoder の文埋め込み比較)の追加分析用コードです。学習は一切行わず、
学習済みモデル・学習途中の checkpoint を推論だけで解析します。狙いは「なぜ・どのように
事前学習済み decoder が適応で伸びるのか」を、attention と埋め込み空間の両面から機構的に
説明する材料を得ることです。

## 何を測るか

`run_geometry_analysis.py` は 1 回のモデルロードで以下をまとめて計算し、`analysis.json` に出力します。

| 分析 | 指標 | 狙い |
| --- | --- | --- |
| attention 要約 | backward mass(未来トークンへの注意割合)、正規化エントロピー、平均注意距離、sink mass(先頭トークン集中)、対角 mass。層別・ヘッド別 | 双方向化(causal → bidirectional)が checkpoint を追ってどの層でどれだけ進むか |
| 位置寄与 | mean pooling への相対位置ビン(10 分割)ごとの cos 寄与・ノルム比 | 「decoder は後方の位置に依存するか」の検証 |
| pooling 変種 | mean / last / first / 位置加重 mean の 4 通りで STS spearman | pooling 選択の寄与分解 |
| outlier 次元 | 次元別分散の top-k・分散シェア・尖度、top-k 次元を 0 化した kill test(STS と alignment/uniformity の再計算) | rogue dimension(Timkey & van Schijndel 2021、Rudman+ 2023)病理の有無 |
| 幾何(層別) | alignment / uniformity(式は既存 isotropic 評価と同一。ただし後述の通り uniformity の対象文集合が異なる)、RankMe・参加率(スペクトル) | 対照学習中の埋め込み空間の幾何変化 |
| 層別 STS | 各層 mean pooling の STS spearman | 文の意味がどの層で立ち上がるか |
| トークン頻度バイアス | log 頻度 vs 最終層トークン表現ノルム(出現平均)の spearman | 頻度依存の異方性(encoder/decoder 差) |
| prefix 頑健性 | 固定 prefix を付けた文と素の文の埋め込み cos | 表層の位置ずれへの安定性 |
| 層別埋め込み dump | 各層 mean pooling 埋め込みの npz(final/base 点のみ) | モデル間の linear CKA・mutual kNN 重複の後段計算用 |

## コード構成

| ファイル | 内容 |
| --- | --- |
| `scripts/constract_llm/eval/analysis/run_geometry_analysis.py` | CLI 本体。pydantic Config + `load_cli_config` + fire(既存 `eval/isotropic/eval.py` と同じ構成) |
| `src/mirei/constract_llm/eval/analysis/metrics.py` | 指標の純関数群(torch/numpy のみ、モデル・IO 非依存) |
| `src/mirei/constract_llm/eval/analysis/loader.py` | モデルロード。ローカル dir に `modules.json` があれば SentenceTransformer 経由、なければ `AutoModel`(`trust_remote_code=True`)。attention 取得のため `attn_implementation='eager'` を強制 |
| `tests/mirei/constract_llm/eval/analysis/test_metrics.py` | 合成テンソルの単体テスト 14 件(causal で backward≈0、一様分布で entropy≈1、rank-1 で RankMe≈1 など) |
| `scripts/constract_llm/eval/analysis/aggregate_results.py` | 全点の `analysis.json` を `aggregate/{scalars,layers}.tsv` へ集計。欠損点・解析不能名・期待点数不一致は既定で例外(`--allow_partial=True` でのみ部分集計) |
| `scripts/constract_llm/eval/analysis/compute_pairwise_cka.py` | final/base 点の `layer_embeddings.npz` から層別 linear CKA と最終層 mutual kNN 重複(既定 `--knn_num_points=1000`、TSV に列として記録)を計算し `aggregate/cka.tsv` へ出力(比較ペアはスクリプト内 `PAIRS` 定数)。ペアの npz 欠損は既定で例外(`--allow_partial=True` でのみ許容) |
| `tests/mirei/constract_llm/eval/analysis/test_bidirectional_mask.py` | 4D mask 経路の回帰テスト(tiny LlamaBiModel を CPU 構築) |
| `config/constract_llm/eval/analysis/example.json` | 実行例 config |

## 重要な設計判断(レビューで特に見てほしい点)

1. **Llama 系 Bi decoder への 4D mask 明示渡し**(`metrics.bidirectional_forward_mask` /
   `metrics.build_model_inputs`、適用判定は `loader.load_backbone`)。
   `LlamaBiModel`(sarashina2.2-Bi を含む Llama 系)の双方向化フック(`_update_causal_mask` の
   override と `self_attn.is_causal=False`)は、transformers 4.56 では FlashAttention-2 経路で
   しか効かず、eager では継承元 `LlamaModel.forward` が causal mask を再構築してしまいます
   (学習・既存評価は FA2 なので影響なし。解析側だけの問題)。そこで **`LlamaBiModel` に限り**
   padding-only の 4D additive mask `(B, 1, 1, T)` を明示的に渡して双方向を保証しています
   (4D mask は transformers 側で as-is で通ります)。Qwen2/Mistral の Bi 実装は独自 `forward` で
   双方向 mask を生成するため対象外です(今回の解析対象の Bi モデルはすべて Llama 系)。
   custom modeling 本体は方針として FA2 経路のみサポートとし、解析側で吸収します。
   この経路は回帰テスト(`test_bidirectional_mask.py`: eager + 2D mask では未来トークンへの
   注意質量が厳密 0、4D mask で正になる)で担保しています。
2. **attention は有効トークン内で再正規化**してから集計(padding の影響を除去)。
   backward/距離は有効トークン内の順位で定義しています。
3. **データソースは既存評価と共通だが、uniformity の対象文集合は異なる**。geometry 文の取得元は
   isotropic 評価と同じ(JA: miracl/miracl + wikimedia/wikipedia 20231101.ja、EN:
   sentence-transformers/all-nli(triplet)+ google/wiki40b、2,000 ペア、seed 42)。ただし既存の
   isotropic 評価は uniformity を **wiki 由来 random pairs の両側(4,000 文)** で計算するのに対し、
   本解析は層別計算の都合上 **正例ペアのアンカー側(2,000 文)** で計算します。式は同一ですが
   対象文集合が異なるため、**本解析の uniformity を論文図5/6 の絶対値と直接比較しないでください**
   (解析内の checkpoint 間・モデル間比較は同一条件です)。alignment は両者とも正例ペアで同一定義。
   STS は EN: GLUE stsb validation、JA: JGLUE JSTS validation の先頭 1,500 ペア。
   attention と位置寄与は 32 文 × 最大 128 トークン。
4. **非有限値は例外で fail**(`_assert_finite`)。黙って NaN を書き出しません。

## 使い方

```bash
python scripts/constract_llm/eval/analysis/run_geometry_analysis.py \
  --config_file_path=config/constract_llm/eval/analysis/example.json

# 引数上書きの例(checkpoint を直接指定)
python scripts/constract_llm/eval/analysis/run_geometry_analysis.py \
  --output_dir=output/ \
  --model_name_or_path=/path/to/run/checkpoint-100 \
  --language=ja \
  --dump_layer_embeddings=True
```

bf16 + eager で GPU 1 枚(A100 で 1 点あたり 2〜6 分)。全点の実行後は CPU のみで集計できます。

```bash
python scripts/constract_llm/eval/analysis/aggregate_results.py \
  --results_root=/path/to/geometry-analysis/<ts> --expected_points=375
python scripts/constract_llm/eval/analysis/compute_pairwise_cka.py \
  --results_root=/path/to/geometry-analysis/<ts>
```

CKA は比較前に、両点の `analysis.json` の設定(language / num_examples / seed / max_seq_length)
一致と、`layer_embeddings.npz` に保存した**エンコード文集合の順序付き SHA-256**(`text_sha256`)の
一致を必須検証します。2026-07-27 実行分の npz はハッシュ導入前の legacy dump のためハッシュを
持ちません。同一 manifest・同一 job で全点が同一設定・同一データ経路で生成されたことを確認済み
のため、この結果に対しては `--allow_unverified_text=True` を明示して実行します(新規 dump からは
フラグ不要)。検証の成否はペアごとに `cka.tsv` の `text_verified` 列へ記録され、provenance が
成果物側に残ります。

## 実行記録(2026-07-27)

- 解析対象 375 点 = 学習 checkpoint **339**(checkpoint 系列が残る **18 run**: WSL 段
  `Sentence-X-PT` と FT 段 `Sentence-X` を別 run と数える、100 step 刻み)+ run 最終 **24**
  (checkpoint 非保存で最終モデルのみの JP 0.5B 系 6 run を含む)+ 双方向化前 base **12**
  (10 モデル、sarashina2.2-Bi は EN/JA 両言語で測定)。
  manifest(`points.tsv`)と sha 検証付き sbatch は
  `pine11:/cl/work16/tatsuki-o/jobs/mirei-geo-20260727/runs/20260727-013310-r1/`。
- **375/375 点 COMPLETED**。出力は
  `/cl/work16/tatsuki-o/data/outputs/mirei-eval/geometry-analysis/20260727-013310/<point>/analysis.json`、
  集計は同 `aggregate/{scalars,layers,cka}.tsv`。
- 実行中に対処した問題:
  - `Sentence-Sarashina-Bi-{EN,JP}-1B-PT` の run root `model.safetensors` が破損
    (保存時 truncation)→ HF の同一モデルへ差し替えて解析(checkpoint と HF は無事)。
  - `Sentence-Llama-Bi-JP-1B-bs1024`(FT 段)の checkpoint が auto_map ローカル参照なのに
    `modeling_bidirectional_llama.py` を同梱していない → source からコピーして解決。

## 検証状況

- 単体テスト **17 件** pass(指標の純関数 14 件 + 4D mask 経路の回帰テスト 3 件。後者は
  tiny `LlamaBiModel` を CPU 構築し「eager + 2D mask では未来への注意質量が厳密 0 /
  4D mask で正」を直接検証)。ruff check / format 済み。フェイクモデルの end-to-end 実行で
  JSON 全セクションを確認。
- クラスタ smoke(縮小サンプル)で encoder final・Bi decoder checkpoint・Bi base の 3 点を実行し、
  期待方向(encoder: backward≈0.45、Bi 修正後: backward>0、base の STS が低い)を確認。
  本文中の修正前後の数値(backward 0.000→0.52、STS 0.45→0.77)は smoke job 536384(修正前)/
  536410(修正後)の実行時観測で、修正前の出力 artifact は削除済みのため再構成できません。
  修正の効果自体は上記回帰テストで再現可能な形で担保しています。
- 既知の制約: 単一 seed・単一データ条件。attention は 32 文のサンプル推定。
  WSL 段の一部 run は `save_total_limit` により序盤 checkpoint が残っておらず、
  軌跡の前半が欠ける(EN 0.5B、JP 1B の ModernBERT/Sarashina)。JP 0.5B は checkpoint 非保存のため
  最終モデル(HF)のみ。

## 暫定的な主要知見(単一 seed・単一データ条件での観察。断定ではなく仮説として)

- 事前学習済み decoder は適応で backward mass が単調に増加(0.46→0.61)するのに対し、
  scratch decoder では増加が観測されなかった(0.47→0.43)。双方向注意の活用度の差が
  事前学習の有無と整合的に現れている。
- STS は WSL 序盤 100 step で急伸(0.39→0.76)し、attention 再編(緩やか)と時間スケールが分離。
- EN 1B の scratch vs 事前学習 decoder: 最終層 CKA 0.98 だが kNN 近傍重複 0.23
  (geometry 文の先頭 1,000 文・k=10 で計算)。当該条件では、事前学習の差は大域幾何よりも
  近傍構造の差として現れている。
- 位置寄与は適応後ほぼフラットで、「decoder は後方依存」を支持する結果は最終モデルでは
  観測されなかった。pooling は decoder では mean が last より大きく上回った(0.82 vs 0.41)。
- rogue dimension の病理は観測されなかった(top1 分散シェア ≤0.4%、kill test ±0.02)。
  attention sink は適応後に減少(0.06→0.02)。頻度-ノルム相関は encoder のみ残存(≈0.25 vs ≈0)。
