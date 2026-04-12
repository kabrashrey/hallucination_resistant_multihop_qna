# hallucination_resistant_multihop_qna

CSCI 566 – Deep Learning Course Project
Evidence-based RAG for Hallucination-Resistant Multi-Hop Q&A

**Description**
Large Language Models (LLMs) often produce hallucinations — factually incorrect or unsupported answers — especially for multi-hop question answering (QA) tasks that require reasoning across multiple documents.

This project aims to build a hallucination-resistant QA pipeline for general-domain multi-hop questions by:

- Grounding answers in retrieved evidence
- Enforcing evidence-first reasoning
- Verifying factual consistency before generating final responses
- Avoiding unsupported generations

We focus primarily on the HotpotQA dataset, which contains multi-hop questions with supporting fact annotations, and optionally explore BeerQA for harder multi-hop settings.

## Repository Structure

```
.
├── README.md
├── configs/
│   ├── default.yaml           # Central YAML config (alpha, rrf_k, hops, paths, etc.)
│   └── prompts.yaml           # Centralized LLM prompts (system/user instructions)
├── data/
│   ├── hotpot_dev_distractor_v1.json
│   ├── hotpot_dev_fullwiki_v1.json
│   └── hotpot_test_fullwiki_v1.json
├── index_cache/               # Saved FAISS + BM25 indices (auto-generated)
├── notebooks/
│   └── baseline_training.ipynb
├── pipeline/
│   ├── __init__.py
│   ├── data_loader.py         # HotpotQA parser with standardized dataclasses
│   ├── indexer.py             # Hybrid multi-hop retriever (BM25 + FAISS)
│   ├── reranker.py            # Re-ranker with sentence-level overlap boosting
│   ├── prompt_builder.py      # Prompt construction and complexity routing
│   ├── generator.py           # Final LLM generation using Ollama
│   └── verifier.py            # Post-generation hallucination verification
├── reports/
├── requirements.txt
├── results/
│   ├── metrics/               # Evaluation metrics outputs
│   └── predictions/           # Model predictions JSONs
└── scripts/
    ├── __init__.py
    ├── config.py              # Typed config loader (YAML → dataclasses)
    ├── analyze_predictions.py # Analysis tools for errors / SP EM scores
    ├── test_pipeline.py       # Unified fast test script for the pipeline
    ├── hotpot_evaluate_v1.py  # Official HotpotQA evaluation script
    └── logger.py              # Colored terminal logger
```

## Setup

1. **Clone the repo:**

   ```bash
   git clone https://github.com/kabrashrey/hallucination_resistant_multihop_qna.git
   cd hallucination_resistant_multihop_qna
   ```

2. **Create and activate a virtual environment:**

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

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Ollama Models:**
   The pipeline assumes a local Ollama instance running. Fetch the required models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.2:1b
   ollama pull qwen3:8b
   ```

## Pipeline Architecture

The pipeline uses a multi-stage process with local, open-source models (via Ollama) and relies on GPU acceleration where available.

### 1. Data Loading & Sub-sampling

`pipeline/data_loader.py` parses HotpotQA JSON into standardized dataclasses (`Passage`, `Context`, `HotpotQAExample`).

### 2. Hybrid Multi-Hop Retriever

`pipeline/indexer.py` combines BM25 (sparse/keyword) and FAISS (dense/semantic) retrieval using Reciprocal Rank Fusion (RRF). Features multi-hop iterative query reformulation and per-question-type alpha dense weights.

### 3. Sentence-level Reranker

`pipeline/reranker.py` reranks retrieved passages based on cross-encoder similarity with exact sentence attribution mapping to compute accurate HotpotQA Supporting Fact Exact Match (SP EM) metrics.

### 4. Dynamic Prompt Routing

`pipeline/prompt_builder.py` builds the final LLM prompt context using an evidence-first approach, and assigns a complexity score to route queries to either a smaller or larger model.

### 5. Final Generator

`pipeline/generator.py` invokes Ollama endpoints seamlessly. Features a **Specialist Mode** and robust parsing mechanisms (via strict schema and timeout configurations).

### 6. Post-Generation Verifier

`pipeline/verifier.py` checks whether the generated answer is actually supported by the retrieved evidence. It decomposes the answer into individual claims and scores each against the evidence sentences. Three verification modes are available:

- **`overlap`** — Pure lexical token overlap (no extra dependencies)
- **`nli`** — Entailment model scoring (65% NLI + 35% lexical), requires `transformers`
- **`qa`** — QA extraction confidence (55% QA + 45% lexical), requires `transformers`

The verifier is integrated into both the eval and test pipelines, producing per-prediction support scores and aggregate diagnostics.

## Usage & Testing

We provide a consolidated end-to-end testing script `test_pipeline.py` which loads a few examples, tests the retrieval, routing, and generation mechanisms.

```bash
python -m scripts.test_pipeline
```

This runs through all 6 stages (retrieve → rerank → prompt build → generate → verify) on a single example and prints the full trace, including verification result.

## Configurations

All system parameters are driven by YAML configurations in `configs/`:

- `default.yaml` - Pipeline logic, thresholds, timeouts, model endpoints, alpha scores.
- `prompts.yaml` - Instruction strings for all stages of retrieval mapping (Standard, Citation, System, User, Specialist).

Example snippet:

```python
from scripts.config import load_config

cfg = load_config("configs/default.yaml")

print(cfg.retriever.alpha)             # 0.7
print(cfg.generator.specialist_mode)   # Feature flag
print(cfg.verifier.enabled)            # True/False
print(cfg.verifier.mode)               # "overlap", "nli", or "qa"
print(cfg.prompts.generator_specialist) # Pull from prompts.yaml
```

## Data & Evaluation

We use **HotpotQA** for multi-hop question answering. Available splits include distractor settings and FullWiki evaluations.
Metrics include macro-averages for Exact Match (EM), F1, Supporting Fact F1, and Joint F1.

Run the full pipeline with verification:

```bash
python -m pipeline.eval --split dev_distractor --limit 250 --eval data/hotpot_dev_distractor_v1.json
```

Run the official evaluation script on saved predictions:

```bash
python scripts/hotpot_evaluate_v1.py results/gold.json results/predictions/predictions.json
```

## Team Members

Ajinkya Nagarkar · Augusto Rivas Constante · Avery Novick · Ishan Chakrabarti · Leonardo Robles-Angeles · Priyanka Rani · Shreyansh Kabra
