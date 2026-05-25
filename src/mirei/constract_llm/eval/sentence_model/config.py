from pydantic import BaseModel, Field


class CLIConfig(BaseModel):
    output_dir: str = Field('results', description='Output directory')
    model_name_or_path: str = Field(..., description='Model name or path to evaluate')
    num_examples: int | None = Field(None, description='Number of samples to use for evaluation (None means all)')
    seed: int = Field(42, description='Random seed for reproducibility')
    miracl_name: str = Field('miracl/miracl', description='HuggingFace dataset name for MIRACL')
    miracl_lang: str = Field('ja', description='Language code for MIRACL')
    wiki_name: str = Field('wikimedia/wikipedia', description='HuggingFace dataset name for Wikipedia')
    wiki_lang: str = Field('20231101.ja', description='Language code or split for Wikipedia')
    positive_pair_dataset_name: str | None = Field(None, description='HF dataset name for positive sentence pairs')
    positive_pair_dataset_config_name: str | None = Field(None, description='HF dataset config for positive pairs')
    positive_pair_dataset_split: str = Field('train', description='HF dataset split for positive pairs')
    positive_pair_sentence1_column: str = Field('anchor', description='First sentence column for positive pairs')
    positive_pair_sentence2_column: str = Field('positive', description='Second sentence column for positive pairs')
