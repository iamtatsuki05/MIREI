# モデル初期化スクリプト

[English](README.md) / 日本語

このディレクトリには、言語モデルの初期化を行うスクリプトが含まれています。

## 概要

`init_model.py`スクリプトは、事前学習済みモデルを初期化し、後続のFineTuneやその他の処理のために準備します。

## 機能

- 様々なモデルタイプをサポート
- Hugging Face Hubからのモデルロードまたはローカルパスからのロード
- モデルをローカルに保存またはHugging Face Hubにプッシュ可能
- 再現性のためのシード設定

## 使用方法

```bash
python scripts/constract_llm/model/init_model.py config/constract_llm/model/init_model/config.json
```

コマンドライン引数を使用して設定ファイルの値を上書きすることも可能です：

```bash
python scripts/constract_llm/model/init_model.py config/constract_llm/model/init_model/config.json --model_type=encoder --push_to_hub=False
```

## 設定ファイル

設定ファイルは`config/constract_llm/model/init_model/config.json`にあります。

### `save_custom_model.py`

FineTune済みモデルやアダプタ構成を標準レイアウトでまとめ、必要に応じて Hub に公開します。

**主な機能**
- `src/mirei/constract_llm/model/save_custom_model.py` の `CUSTOM_MODEL_CONFIGS` で定義されたカスタムモデルをサポート。
- タスク種別（例：`mntp`, `st`）と必須メタデータを検証した上でエクスポート。
- モデルとトークナイザーを指定ディレクトリに保存し、プライベート／パブリックの Hub リポジトリへプッシュ可能。

**使用方法**

```bash
python scripts/constract_llm/model/save_custom_model.py config/constract_llm/model/save_custom_model/llama_bi.json
python scripts/constract_llm/model/save_custom_model.py --model_name_or_path path/to/model --custom_model_type llama_bi --task_type mntp --output_dir ./artifacts
```

設定例は `config/constract_llm/model/save_custom_model/` に用意されています。`push_to_hub` を有効にする場合は `repo_id` の指定を忘れないでください。
