# アライメント・ユニフォーミティ評価スクリプト

このディレクトリには、[SentenceTransformer](https://www.sbert.net/) ライブラリを用いて文埋め込みモデルのアライメント（alignment）およびユニフォーミティ（uniformity）指標を評価するスクリプトが含まれています。

## 概要

`eval.py` は、指定した文埋め込みモデルに対して以下の指標を計算します。
- **アライメント（Alignment）**: 正例ペアが埋め込み空間でどれだけ近いかを測定
- **ユニフォーミティ（Uniformity）**: 埋め込みが高次元球面上でどれだけ一様に分布しているかを測定

設定ファイル（JSON/YAML/TOML）またはコマンドライン引数で柔軟にパラメータ指定が可能です。

## 使い方

```bash
# アライメントとユニフォーミティを両方計算
python scripts/constract_llm/eval/isotropic/eval.py main --config_file_path=config/constract_llm/eval/isotropic/example.json

# アライメントのみ計算
python scripts/constract_llm/eval/isotropic/eval.py alignment --config_file_path=config/constract_llm/eval/isotropic/example.json

# ユニフォーミティのみ計算
python scripts/constract_llm/eval/isotropic/eval.py uniformity --config_file_path=config/constract_llm/eval/isotropic/example.json
```

コマンドライン引数で設定値を上書きすることも可能です。

```bash
python scripts/constract_llm/eval/isotropic/eval.py main --model_name_or_path=sentence-transformers/all-MiniLM-L6-v2 --output_dir=output/
```

## 設定ファイル

設定ファイルは `pydantic.BaseModel` 互換（`src/mirei/constract_llm/eval/isotropic/config.py` 参照）です。
例（JSON）:

```json
{
  "model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2",
  "output_dir": "output/",
  "num_examples": 1000,
  "seed": 42,
  "miracl_name": "miracl-ja",
  "miracl_lang": "ja",
  "wiki_name": "wikipedia-ja",
  "wiki_lang": "ja"
}
```

## 出力

結果は以下のパスにJSON形式で保存されます。

```
<output_dir>/alignment_and_uniformity/<model_name_or_path>/result.json
```

出力例:

```json
{
  "alignment": 0.1234,
  "uniformity": 1.2345
}
```
