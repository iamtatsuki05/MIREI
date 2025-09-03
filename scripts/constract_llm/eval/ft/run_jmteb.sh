git clone https://github.com/sbintuitions/JMTEB.git
cd JMTEB
# https://github.com/Lightning-AI/litgpt/issues/1915
rm poetry.lock
poetry install

poetry run python -m jmteb \
    --embedder DataParallelSentenceBertEmbedder \
    --embedder.model_name_or_path "iamtatsuki05/Sentence-Sarashina-Bi-0.5B" \
    --save_dir "output/Sentence-Sarashina-Bi-0.5B"


poetry run python -m jmteb \
    --embedder DataParallelSentenceBertEmbedder \
    --embedder.model_name_or_path "iamtatsuki05/Sentence-ModernBERT-JP-0.5B" \
    --save_dir "output/Sentence-ModernBERT-JP-0.5B"


poetry run python -m jmteb \
    --embedder DataParallelSentenceBertEmbedder \
    --embedder.model_name_or_path "iamtatsuki05/Sentence-Llama-Bi-JP-0.5B" \
    --save_dir "output/Sentence-Llama-Bi-JP-0.5B"
