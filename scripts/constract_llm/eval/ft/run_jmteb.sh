git clone https://github.com/sbintuitions/JMTEB.git
cd JMTEB
# https://github.com/Lightning-AI/litgpt/issues/1915
poetry install
poetry add "jsonargparse>=4.36" jsonnet "peft==0.1.0" "accelerate>=1.10.1" "transformers==4.51.0" "protobuf>=3.20.1" sentencepiece
poetry update jsonargparse jsonnet accelerate transformers protobuf sentencepiece

poetry run python -m jmteb \
    --embedder SentenceBertEmbedder \
    --embedder.model_name_or_path "iamtatsuki05/Sentence-Sarashina-Bi-0.5B" \
    --save_dir "output/Sentence-Sarashina-Bi-0.5B"


poetry run python -m jmteb \
    --embedder SentenceBertEmbedder \
    --embedder.model_name_or_path "iamtatsuki05/Sentence-ModernBERT-JP-0.5B" \
    --save_dir "output/Sentence-ModernBERT-JP-0.5B"


poetry run python -m jmteb \
    --embedder SentenceBertEmbedder \
    --embedder.model_name_or_path "iamtatsuki05/Sentence-Llama-Bi-JP-0.5B" \
    --save_dir "output/Sentence-Llama-Bi-JP-0.5B"
