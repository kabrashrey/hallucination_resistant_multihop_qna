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
.
├── README.md
├── configs/                        # Configuration files
├── data/                           # HotpotQA dataset files
├── notebooks/                      # EDA, experiments, and analysis notebooks
├── pipeline/                       # Core RAG pipeline implementation
├── requirements.txt                # Python dependencies
├── results/                        # Model outputs and evaluation results
└── scripts/
    └── hotpot_evaluate_v1.py       # Official HotpotQA evaluation script

## Setup
1. Clone the repo: 
git clone https://github.com/kabrashrey/hallucination_resistant_multihop_qna.git

2. Create and activate a virtual environment:

Using **venv** : 
python -m venv .venv
source .venv/bin/activate

or Using **conda** :
conda create --name rag_multihop python=3.10
conda activate rag_multihop

3. Install dependencies 
pip install -r requirements.txt


## Configurations
All experiment parameters (retriever type, top-k, model name, etc.) should be defined inside the configs/ directory.


## Data
We use HotpotQA for multi-hop question answering.

Available splits:
    hotpot_dev_distractor_v1.json
    hotpot_dev_fullwiki_v1.json
    hotpot_test_fullwiki_v1.json

Two evaluation settings:
    Distractor Setting – Smaller candidate document set (easier)
    FullWiki Setting – Requires full corpus retrieval (harder, realistic)


## Pipeline Overview
The system follows a modular RAG-based architecture:
1. Retrieval
2. Evidence Processing
3. Verification

## Models

## Evaluation
Metrics include:
    Exact Match (EM)
    F1 

## Team Members
Ajinkya Nagarkar · Augusto Rivas Constante · Avery Novick · Ishan Chakrabarti · Leonardo Robles-Angeles · Priyanka Rani · Shreyansh Kabra
