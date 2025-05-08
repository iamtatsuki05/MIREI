git clone https://github.com/sbintuitions/JMTEB.git
cd JMTEB
# https://github.com/Lightning-AI/litgpt/issues/1915
rm poetry.lock
poetry install

poetry run python -m jmteb \
    --embedder SentenceBertEmbedder \
    --embedder.model_name_or_path "iamtatsuki05/Sentence-ModernBERT-JP-1.4B" \
    --save_dir "output/Sentence-ModernBERT-JP-1.4B"
