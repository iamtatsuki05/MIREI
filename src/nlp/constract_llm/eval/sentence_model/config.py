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
