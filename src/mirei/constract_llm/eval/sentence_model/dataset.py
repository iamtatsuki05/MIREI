from datasets import load_dataset


def prepare_dataset(
    num_examples: int | None = None,
    miracl_name: str = 'miracl/miracl',
    miracl_lang: str = 'ja',
    wiki_name: str = 'wikimedia/wikipedia',
    wiki_lang: str = '20231101.ja',
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Prepare positive and random sentence pairs for evaluation.

    Args:
        num_examples: Number of positive/random pairs to generate (None means all)
        miracl_name: HuggingFace dataset name for MIRACL
        miracl_lang: Language code for MIRACL
        wiki_name: HuggingFace dataset name for Wikipedia
        wiki_lang: Language code or split for Wikipedia

    Returns:
        tuple of (positive_pairs, random_pairs)
        - positive_pairs: list of (sentence1, sentence2) from the same context
        - random_pairs: list of (sentence1, sentence2) from different contexts
    """
    miracl_ds = load_dataset(miracl_name, miracl_lang, trust_remote_code=True, split='train')
    wiki_ds = load_dataset(wiki_name, wiki_lang, split='train').shuffle(seed=42)
    positive_pairs = []
    for data in miracl_ds:
        positive_passages = data['positive_passages']
        positive_sentences = [entry['text'] for entry in positive_passages]
        if len(positive_sentences) < 2:
            continue
        positive_pairs.append((positive_sentences[0], positive_sentences[1]))

    if num_examples is not None:
        positive_pairs = positive_pairs[:num_examples]
        wiki_limit = num_examples * 2
    else:
        wiki_limit = len(positive_pairs) * 2

    corpus_sentences = []
    for i, example in enumerate(wiki_ds):
        if num_examples is not None and i >= wiki_limit:
            break
        corpus_sentences.append(example['text'])
    random_pairs = []
    n_pairs = num_examples if num_examples is not None else len(positive_pairs)
    for i in range(n_pairs):
        random_pairs.append((corpus_sentences[i], corpus_sentences[n_pairs + i]))
    return positive_pairs, random_pairs
