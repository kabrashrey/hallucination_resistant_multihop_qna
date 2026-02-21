# hallucination_resistant_multihop_qna

CSCI 566 – Deep Learning Course Project
Evidence-based RAG for Hallucination-Resistant Multi-Hop Q&A

**Description**
Large Language Models (LLMs) often produce hallucinations — factually incorrect or unsupported answers — especially for multi-hop question answering (QA) tasks that require reasoning across multiple documents.

This project aims to build a hallucination-resistant QA pipeline for general-domain multi-hop questions by:
Grounding answers in retrieved evidence
Enforcing evidence-first reasoning
Verifying factual consistency before generating final responses
Avoiding unsupported generations

We focus primarily on the HotpotQA dataset, which contains multi-hop questions with supporting fact annotations, and optionally explore BeerQA for harder multi-hop settings.

## Repository Structure

```
.
├── README.md
├── configs/
│   └── default.yaml          # Central YAML config (alpha, rrf_k, hops, paths, etc.)
├── data/
│   ├── hotpot_dev_distractor_v1.json
│   ├── hotpot_dev_fullwiki_v1.json
│   └── hotpot_test_fullwiki_v1.json
├── index_cache/               # Saved FAISS + BM25 indices (auto-generated)
├── notebooks/
│   └── baseline_training.ipynb
├── pipeline/
│   ├── __init__.py
│   ├── config.py              # Typed config loader (YAML → dataclasses)
│   ├── data_loader.py         # HotpotQA parser with standardized dataclasses
│   ├── indexer.py             # Hybrid multi-hop retriever (BM25 + FAISS)
│   └── logger.py              # Colored terminal logger
├── reports/
├── requirements.txt
├── results/                   # Predictions and evaluation outputs
└── scripts/
    └── hotpot_evaluate_v1.py  # Official HotpotQA evaluation script
```

## Setup

1. Clone the repo:

   ```bash
   git clone https://github.com/kabrashrey/hallucination_resistant_multihop_qna.git
   ```

2. Create and activate a virtual environment:

   Using **venv**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   or Using **conda**:

   ```bash
   conda create --name rag_multihop python=3.10
   conda activate rag_multihop
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Pipeline Modules

### `pipeline/data_loader.py` — Data Loading

Parses HotpotQA JSON into standardized dataclasses (`Passage`, `SupportingFact`, `HotpotQAExample`).

```python
from pipeline.data_loader import HotpotQALoader

loader = HotpotQALoader("data/hotpot_dev_distractor_v1.json")
examples = loader.load(limit=100)
passages = list(loader.build_passage_index().values())
```

```bash
python -m pipeline.data_loader  data/hotpot_dev_distractor_v1.json 3
```

### `pipeline/indexer.py` — Hybrid Multi-Hop Retriever

Combines BM25 (sparse, keyword) and FAISS (dense, semantic) retrieval with Reciprocal Rank Fusion (RRF).

**Features:**

- **Candidate-pool RRF** — only fuses top-N from each retriever, not the full corpus
- **Per-type alpha** — different dense/BM25 weights for bridge vs comparison questions
- **Multi-hop retrieval** — iterative query reformulation with bridge entity extraction
- **Confidence signal** — per-hop score gap between #1 and #2
- **Save/load** — persist FAISS + BM25 indices to disk, skip re-encoding

```python
from pipeline.indexer import HybridRetriever

retriever = HybridRetriever.from_config()  # reads configs/default.yaml
retriever.index(passages)
retriever.save("index_cache")

# Single-hop
results = retriever.retrieve("Who directed Doctor Strange?", top_k=5)

# Multi-hop (bridge questions)
results = retriever.retrieve_multihop("What position was held by the woman who portrayed Corliss Archer?", hops=2, top_k=5, question_type="bridge")
```

```bash
python -m pipeline.indexer
```

### `pipeline/config.py` — Configuration

All tunable parameters live in `configs/default.yaml`. Load with typed dataclasses:

```python
from pipeline.config import load_config

cfg = load_config()                           # default config
cfg = load_config("configs/experiment1.yaml") # custom config

cfg.retriever.alpha           # 0.7
cfg.retriever.alpha_bridge    # 0.85
cfg.retriever.multihop.hops   # 2
cfg.data.dev_distractor       # "data/hotpot_dev_distractor_v1.json"
```

### `pipeline/logger.py` — Colored Logger

Colored terminal output (green info/success, red warning/error, yellow step, gray debug):

```python
from pipeline.logger import get_logger
log = get_logger("my_module")

log.info("Processing...")      # green
log.success("Done!")           # green bold
log.warning("Slow operation")  # red
log.error("Failed!")           # red bold
```

## Configurations

All experiment parameters are in `configs/default.yaml`:

| Parameter                          | Default          | Description                             |
| ---------------------------------- | ---------------- | --------------------------------------- |
| `retriever.alpha`                  | 0.7              | Dense weight (1-alpha = BM25 weight)    |
| `retriever.alpha_bridge`           | 0.85             | Alpha for bridge questions              |
| `retriever.alpha_comparison`       | 0.5              | Alpha for comparison questions          |
| `retriever.rrf_k`                  | 20               | RRF smoothing constant                  |
| `retriever.candidate_pool_size`    | 100              | Top-N from each retriever before fusion |
| `retriever.embed_model`            | all-MiniLM-L6-v2 | SentenceTransformer model               |
| `retriever.multihop.hops`          | 2                | Retrieval iterations                    |
| `retriever.multihop.top_k_per_hop` | 5                | Passages per hop                        |

## Data

We use HotpotQA for multi-hop question answering.

Available splits:

- `hotpot_dev_distractor_v1.json` — smaller candidate set (easier, 10 passages per question)
- `hotpot_dev_fullwiki_v1.json` — full corpus retrieval (harder, realistic)
- `hotpot_test_fullwiki_v1.json` — test set (no gold answers)

Two evaluation settings:

- **Distractor Setting** — 10 candidate passages per question
- **FullWiki Setting** — requires retrieval from the full corpus

## Evaluation

Metrics include:

- Exact Match (EM)
- F1
- Supporting Fact EM / F1
- Joint EM / F1

Run evaluation:

```bash
python scripts/hotpot_evaluate_v1.py results/predictions.json results/gold.json
```

## Team Members

Ajinkya Nagarkar · Augusto Rivas Constante · Avery Novick · Ishan Chakrabarti · Leonardo Robles-Angeles · Priyanka Rani · Shreyansh Kabra
