"""
Config Loader
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class DataConfig:
    dev_distractor: str = "data/hotpot_dev_distractor_v1.json"
    dev_fullwiki: str = "data/hotpot_dev_fullwiki_v1.json"
    test_fullwiki: str = "data/hotpot_test_fullwiki_v1.json"
    limit: Optional[int] = None


@dataclass
class MultihopConfig:
    hops: int = 2
    top_k_per_hop: int = 5
    max_bridge_entities: int = 3
    hop2_strategy: str = "both"                          # "title" | "concat" | "both"
    llm_decompose_confidence_threshold: float = 0.3      # trigger LLM decomposition below this
    llm_decompose_ollama_model: str = "qwen2.5:7b"
    extraction_temperature: float = 0.0                  # LLM temperature for bridge entity extraction
    extraction_timeout: int = 120                        # timeout for local LLM during entity extraction


@dataclass
class RetrieverConfig:
    embed_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"
    device: str = "cpu"
    batch_size: int = 64
    alpha: float = 0.7
    alpha_bridge: Optional[float] = 0.85
    alpha_comparison: Optional[float] = 0.5
    rrf_k: int = 20
    candidate_pool_size: int = 100
    top_k: int = 10
    multihop: MultihopConfig = field(default_factory=MultihopConfig)
    index_cache_dir: str = "index_cache"


@dataclass
class RerankerConfig:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    sentence_model_name: str = "nomic-embed-text"
    device: str = "cpu"
    top_k: int = 5                        
    sentence_score_threshold: float = 0.4
    max_sentences_per_passage: int = 5
    batch_size: int = 32
    sentence_passage_limit: int = 3
    title_overlap_boost: float = 0.05


@dataclass
class PromptBuilderConfig:
    # Formatting options
    include_passage_numbers: bool = True       
    include_sentence_indices: bool = True      
    evidence_first: bool = True                
    max_evidence_chars: int = 12000             

    complexity_length_weight: float = 0.25
    complexity_length_threshold: int = 50
    complexity_keywords_weight: float = 0.25
    complexity_confidence_weight: float = 0.25
    complexity_sentences_weight: float = 0.25
    complexity_sentence_threshold: int = 5     
    complexity_routing_threshold: float = 0.6  

    temperature_small_model: float = 0.2       
    temperature_large_model: float = 0.4       


@dataclass
class GeneratorConfig:
    ollama_base_url: str = "http://localhost:11434"
    model_small: str = "mistral:7b"
    model_large: str = "mistral:7b"
    request_timeout: int = 300
    validate_citations: bool = True
    retry_on_parse_failure: bool = True
    specialist_mode: bool = False                       # True: model_large (SP) + model_small (answer)


@dataclass
class EvalConfig:
    limit: Optional[int] = 100
    predictions_dir: str = "results"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    prompt_builder: PromptBuilderConfig = field(default_factory=PromptBuilderConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _dict_to_dataclass(cls, d: dict):
    if d is None:
        return cls()
    fieldtypes = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for key, val in d.items():
        if key in fieldtypes:
            ft = fieldtypes[key]
            # Check if the field type is itself a dataclass
            if isinstance(ft, type) and hasattr(ft, '__dataclass_fields__') and isinstance(val, dict):
                kwargs[key] = _dict_to_dataclass(ft, val)
            else:
                kwargs[key] = val
    return cls(**kwargs)


def load_config(path: Union[str, Path, None] = None) -> Config:
    """
    Load config from YAML file
    path: path to YAML file (default: configs/default.yaml)
    Returns: Config dataclass
    """
    if path is None:
        # Look for default.yaml relative to project root
        path = Path("configs/default.yaml")

    path = Path(path)
    if not path.exists():
        from scripts.logger import get_logger
        get_logger("config").warning(f"Config not found at {path}, using defaults")
        return Config()

    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    cfg = Config()

    if "data" in raw:
        cfg.data = _dict_to_dataclass(DataConfig, raw["data"])

    if "retriever" in raw:
        ret = raw["retriever"]
        multihop_raw = ret.pop("multihop", None)
        cfg.retriever = _dict_to_dataclass(RetrieverConfig, ret)
        if multihop_raw:
            cfg.retriever.multihop = _dict_to_dataclass(MultihopConfig, multihop_raw)

    if "reranker" in raw:
        cfg.reranker = _dict_to_dataclass(RerankerConfig, raw["reranker"])

    if "prompt_builder" in raw:
        cfg.prompt_builder = _dict_to_dataclass(PromptBuilderConfig, raw["prompt_builder"])

    if "generator" in raw:
        cfg.generator = _dict_to_dataclass(GeneratorConfig, raw["generator"])

    if "eval" in raw:
        cfg.eval = _dict_to_dataclass(EvalConfig, raw["eval"])

    return cfg


if __name__ == "__main__":
    cfg = load_config()
    from scripts.logger import get_logger
    log = get_logger("config")
    log.info(f"Data:")
    log.info(f"  distractor: {cfg.data.dev_distractor}")
    log.info(f"  fullwiki:   {cfg.data.dev_fullwiki}")
    log.info(f"  limit:      {cfg.data.limit}")
    log.info(f"Retriever:")
    log.info(f"  model:      {cfg.retriever.embed_model}")
    log.info(f"  alpha:      {cfg.retriever.alpha}")
    log.info(f"  bridge:     {cfg.retriever.alpha_bridge}")
    log.info(f"  comparison: {cfg.retriever.alpha_comparison}")
    log.info(f"  rrf_k:      {cfg.retriever.rrf_k}")
    log.info(f"  pool_size:  {cfg.retriever.candidate_pool_size}")
    log.info(f"  top_k:      {cfg.retriever.top_k}")
    log.info(f"  cache:      {cfg.retriever.index_cache_dir}")
    log.info(f"Multi-hop:")
    log.info(f"  hops:       {cfg.retriever.multihop.hops}")
    log.info(f"  per_hop:    {cfg.retriever.multihop.top_k_per_hop}")
    log.info(f"  entities:   {cfg.retriever.multihop.max_bridge_entities}")
    log.info(f"Eval:")
    log.info(f"  limit:      {cfg.eval.limit}")
    log.info(f"  output_dir: {cfg.eval.predictions_dir}")
