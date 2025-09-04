import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
 
model = AutoModelForCausalLM.from_pretrained("iamtatsuki05/Llama-JP-0.5B-PT-stage1", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("iamtatsuki05/Llama-JP-0.5B-PT-stage1")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
set_seed(123)
 
text = generator(
    "おはようございます、今日の天気は",
    max_length=30,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id,
    num_return_sequences=3,
)

for t in text:
    print(t)
