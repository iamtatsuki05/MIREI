git clone https://github.com/sbintuitions/JMTEB.git
cd JMTEB
poetry install
poetry run pip install flash-attn --no-build-isolation

poetry run python -m jmteb \
    --embedder SentenceBertEmbedder \
    --embedder.model_name_or_path "iamtatsuki05/Sentence-Sarashina-Bi-0.5B" \
    --embedder.device cuda \
    --embedder.model_kwargs '{"torch_dtype": "torch.bfloat16", "attn_implementation": "flash_attention_2"}' \
    --embedder.max_seq_length 2048 \
    --embedder.batch_size 512 \
    --save_dir "output/Sentence-Sarashina-Bi-0.5B/len2048"


poetry run python -m jmteb \
    --embedder SentenceBertEmbedder \
    --embedder.model_name_or_path "iamtatsuki05/Sentence-ModernBERT-JP-0.5B" \
    --embedder.device cuda \
    --embedder.model_kwargs '{"torch_dtype": "torch.bfloat16", "attn_implementation": "flash_attention_2"}' \
    --embedder.max_seq_length 2048 \
    --embedder.batch_size 512 \
    --save_dir "output/Sentence-ModernBERT-JP-0.5B/len2048"


poetry run python -m jmteb \
    --embedder SentenceBertEmbedder \
    --embedder.model_name_or_path "iamtatsuki05/Sentence-Llama-Bi-JP-0.5B" \
    --embedder.device cuda \
    --embedder.model_kwargs '{"torch_dtype": "torch.bfloat16", "attn_implementation": "flash_attention_2"}' \
    --embedder.max_seq_length 2048 \
    --embedder.batch_size 512 \
    --save_dir "output/Sentence-Llama-Bi-JP-0.5B/len2048"


poetry run python -m jmteb \
    --embedder SentenceBertEmbedder \
    --embedder.model_name_or_path "iamtatsuki05/Sentence-Sarashina-Bi-0.5B" \
    --embedder.device cuda \
    --embedder.model_kwargs '{"torch_dtype": "torch.bfloat16", "attn_implementation": "flash_attention_2"}' \
    --embedder.max_seq_length 512 \
    --embedder.batch_size 1024 \
    --save_dir "output/Sentence-Sarashina-Bi-0.5B/len512"


poetry run python -m jmteb \
    --embedder SentenceBertEmbedder \
    --embedder.model_name_or_path "iamtatsuki05/Sentence-ModernBERT-JP-0.5B" \
    --embedder.device cuda \
    --embedder.model_kwargs '{"torch_dtype": "torch.bfloat16", "attn_implementation": "flash_attention_2"}' \
    --embedder.max_seq_length 512 \
    --embedder.batch_size 1024 \
    --save_dir "output/Sentence-ModernBERT-JP-0.5B/len512"


poetry run python -m jmteb \
    --embedder SentenceBertEmbedder \
    --embedder.model_name_or_path "iamtatsuki05/Sentence-Llama-Bi-JP-0.5B" \
    --embedder.device cuda \
    --embedder.model_kwargs '{"torch_dtype": "torch.bfloat16", "attn_implementation": "flash_attention_2"}' \
    --embedder.max_seq_length 512 \
    --embedder.batch_size 1024 \
    --save_dir "output/Sentence-Llama-Bi-JP-0.5B/len512"
