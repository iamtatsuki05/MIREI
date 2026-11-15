from sentence_transformers import SentenceTransformer, models


def replace_pooling(model: SentenceTransformer, pooling_mode: str) -> None:
    # Only the default [Transformer, Pooling] stack built from a plain HF checkpoint is supported; anything else
    # (e.g. a saved SentenceTransformer with extra modules) would be silently mis-pooled, so fail fast.
    if len(model) != 2 or not isinstance(model[1], models.Pooling):
        raise ValueError(
            f'pooling_mode requires a [Transformer, Pooling] model, got {[type(m).__name__ for m in model]}'
        )
    model[1] = models.Pooling(model[1].get_sentence_embedding_dimension(), pooling_mode=pooling_mode)


def check_special_tokens(model: SentenceTransformer, add_bos_token: bool | None, add_eos_token: bool | None) -> None:
    # Some tokenizer classes accept add_bos_token / add_eos_token and silently ignore them, which would make cls or
    # lasttoken pooling read a content token instead of the special token.
    ids = model.tokenize(['a'])['input_ids'][0].tolist()
    tokenizer = model.tokenizer
    if add_bos_token and ids[0] != tokenizer.bos_token_id:
        raise ValueError(f'add_bos_token=True but the tokenized input does not start with BOS: {ids}')
    if add_eos_token and ids[-1] != tokenizer.eos_token_id:
        raise ValueError(f'add_eos_token=True but the tokenized input does not end with EOS: {ids}')
