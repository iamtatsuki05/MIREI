# **MIREI**: **M**atched **I**nvestigation of **R**epresentation **E**mbedding **I**nsights

English / [日本語](README_JA.md)

## uvの操作方法
### セットアップ
1. `git clone`でインストール
### uv設定
1. `uv sync`
### スクリプト実行
```shell
uv run python ...
```

## Dockerの操作方法
### セットアップ
1. `git clone`でインストール
### Docker設定
1. `docker compose up -d --build <サービス名(例:python-cpu)>`
### Dockerへの接続と切断
1. 接続：`docker compose exec <サービス名(例:python-cpu)> bash`
2. 切断：`exit`
### JupyterLabの使用
1. ブラウザでアクセス http://localhost:8888/lab
### コンテナの起動と停止
1. 起動：`docker compose start`
2. 停止：`docker compose stop`

## ディレクトリ構造
```text
./
├── .dockerignore
├── .git
├── .gitattributes
├── .github
├── .gitignore
├── .pre-commit-config.yaml
├── Makefile
├── README.md
├── README_JA.md
├── compose.yaml
├── config
├── data
│   ├── datasets
│   ├── misc
│   ├── models
│   ├── outputs
│   └── raw
├── docker
│   ├── cpu
│   └── gpu
├── docs
├── env.sample
├── notebooks
├── uv.lock
├── pyproject.toml
├── scripts
│   ├── main.py
│   ├── README.md
│   ├── README_JA.md
│   └── constract_llm
│       ├── README.md
│       ├── README_JA.md
│       ├── dataset
│       ├── model
│       ├── tokenizer
│       └── train
│           ├── README.md
│           ├── README_JA.md
│           ├── ft
│           └── pt
├── src
│   ├── __init__.py
│   └── nlp
│       ├── common
│       ├── config
│       ├── env.py
│       └── constract_llm
└── tests
    └── nlp
```

## スクリプト

このプロジェクトには、言語モデル（LLM）の構築と訓練に関連する様々なスクリプトが含まれています。詳細については、以下のREADMEを参照してください：

- [スクリプト一覧](scripts/README_JA.md) - 基本的なスクリプトの概要
- [言語モデル構築スクリプト](scripts/constract_llm/README_JA.md) - 言語モデル構築に関連するスクリプト
- [学習スクリプト](scripts/constract_llm/train/README_JA.md) - 事前学習とFTのスクリプト
  - [事前学習スクリプト](scripts/constract_llm/train/pt/README_JA.md) - MLMとMNTPの事前学習スクリプト
  - [FTスクリプト](scripts/constract_llm/train/ft/README_JA.md) - Sentence TransformerFTスクリプト
