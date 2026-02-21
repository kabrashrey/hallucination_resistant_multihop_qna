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


@dataclass
class RetrieverConfig:
    embed_model: str = "all-MiniLM-L6-v2"
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
class EvalConfig:
    limit: Optional[int] = 100
    predictions_dir: str = "results"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
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
        from pipeline.logger import get_logger
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

    if "eval" in raw:
        cfg.eval = _dict_to_dataclass(EvalConfig, raw["eval"])

    return cfg


if __name__ == "__main__":
    cfg = load_config()
    from pipeline.logger import get_logger
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
