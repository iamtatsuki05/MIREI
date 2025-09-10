git clone -b feat/update_datasets https://github.com/sbintuitions/JMTEB.git # ver-2 if deleted this branch you can use main branch
cd JMTEB
# https://github.com/Lightning-AI/litgpt/issues/1915
poetry install
poetry add "jsonargparse>=4.36" jsonnet "peft==0.1.0" "accelerate>=1.10.1" "transformers==4.51.0" "protobuf>=3.20.1" sentencepiece "torch>=2.6.0"
poetry update jsonargparse jsonnet accelerate transformers protobuf sentencepiece torch

poetry run python -m jmteb \
    --embedder SentenceBertEmbedder \
    --embedder.model_name_or_path "iamtatsuki05/Sentence-Sarashina-Bi-0.5B" \
    --embedder.device cuda \
    --embedder.model_kwargs '{"torch_dtype": "torch.bfloat16"}' \
    --embedder.max_seq_length 2048 \
    --embedder.batch_size 512 \
    --save_dir "output/Sentence-Sarashina-Bi-0.5B/len2048"


poetry run python -m jmteb \
    --embedder SentenceBertEmbedder \
    --embedder.model_name_or_path "iamtatsuki05/Sentence-ModernBERT-JP-0.5B" \
    --embedder.device cuda \
    --embedder.model_kwargs '{"torch_dtype": "torch.bfloat16"}' \
    --embedder.max_seq_length 2048 \
    --embedder.batch_size 512 \
    --save_dir "output/Sentence-ModernBERT-JP-0.5B/len2048"


poetry run python -m jmteb \
    --embedder SentenceBertEmbedder \
    --embedder.model_name_or_path "iamtatsuki05/Sentence-Llama-Bi-JP-0.5B" \
    --embedder.device cuda \
    --embedder.model_kwargs '{"torch_dtype": "torch.bfloat16"}' \
    --embedder.max_seq_length 2048 \
    --embedder.batch_size 512 \
    --save_dir "output/Sentence-Llama-Bi-JP-0.5B/len2048"


poetry run python -m jmteb \
    --embedder SentenceBertEmbedder \
    --embedder.model_name_or_path "iamtatsuki05/Sentence-Sarashina-Bi-0.5B" \
    --embedder.device cuda \
    --embedder.model_kwargs '{"torch_dtype": "torch.bfloat16"}' \
    --embedder.max_seq_length 512 \
    --embedder.batch_size 512 \
    --save_dir "output/Sentence-Sarashina-Bi-0.5B/len512"


poetry run python -m jmteb \
    --embedder SentenceBertEmbedder \
    --embedder.model_name_or_path "iamtatsuki05/Sentence-ModernBERT-JP-0.5B" \
    --embedder.device cuda \
    --embedder.model_kwargs '{"torch_dtype": "torch.bfloat16"}' \
    --embedder.max_seq_length 512 \
    --embedder.batch_size 512 \
    --save_dir "output/Sentence-ModernBERT-JP-0.5B/len512"


poetry run python -m jmteb \
    --embedder SentenceBertEmbedder \
    --embedder.model_name_or_path "iamtatsuki05/Sentence-Llama-Bi-JP-0.5B" \
    --embedder.device cuda \
    --embedder.model_kwargs '{"torch_dtype": "torch.bfloat16"}' \
    --embedder.max_seq_length 512 \
    --embedder.batch_size 512 \
    --save_dir "output/Sentence-Llama-Bi-JP-0.5B/len512"
