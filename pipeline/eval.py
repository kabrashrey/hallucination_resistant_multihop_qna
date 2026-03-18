"""Pipeline: DataLoader → HybridRetriever → Reranker → PromptBuilder → Generator"""
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Optional

from tqdm import tqdm

from pipeline.data_loader import HotpotQALoader
from pipeline.indexer import HybridRetriever
from pipeline.reranker import Reranker
from pipeline.prompt_builder import PromptBuilder
from pipeline.generator import Generator
from scripts.config import load_config
from scripts.logger import get_logger

log = get_logger("eval")


def build_pipeline(cfg):
    log.info("Loading data...")
    loader = HotpotQALoader(cfg.data.dev_distractor)
    examples = loader.load(limit=cfg.eval.limit)

    seen = set()
    passages = []
    for ex in examples:
        for ctx in ex.contexts:
            key = (ctx.title, ctx.text)
            if key not in seen:
                seen.add(key)
                passages.append(ctx)

    log.info(f"Building retriever with {len(passages)} passages...")
    retriever = HybridRetriever.from_config(cfg)
    retriever.index(passages, show_progress=False)

    log.info("Loading reranker...")
    reranker = Reranker.from_config(cfg)

    log.info("Initializing prompt builder...")
    pb = PromptBuilder.from_config(cfg)

    log.info("Initializing generator...")
    gen = Generator.from_config(cfg)

    return examples, retriever, reranker, pb, gen, cfg


def run_pipeline(examples, retriever, reranker, pb, gen, cfg, limit: Optional[int] = None):
    predictions = {}

    if limit:
        examples = examples[:limit]

    # Stage-by-stage recall diagnostics
    diag = {"retrieval": [], "rerank": [], "sentence": [], "prediction": []}

    reranker_top_k = getattr(cfg.reranker, 'top_k', 7)

    for ex in tqdm(examples, desc="Running pipeline"):
        # Gold supporting facts for this example
        gold_titles = {sf.title for sf in ex.supporting_facts}
        gold_facts = {(sf.title, sf.sentence_index) for sf in ex.supporting_facts}

        retrieved = retriever.retrieve_multihop(
            ex.question, hops=2, top_k=20, question_type=ex.question_type
        )

        if not retrieved:
            predictions[ex.id] = {
                "answer": "Cannot answer based on the provided evidence.",
                "sp": [],
            }
            for stage in diag:
                diag[stage].append(0.0)
            continue

        # Diagnostic: retrieval recall (gold titles in retrieved candidates)
        retrieved_titles = {r.passage.title for r in retrieved}
        retrieval_recall = len(gold_titles & retrieved_titles) / len(gold_titles) if gold_titles else 0.0
        diag["retrieval"].append(retrieval_recall)

        reranked = reranker.rerank(ex.question, retrieved, top_k=reranker_top_k, select_sentences=True)

        # Diagnostic: rerank recall (gold titles surviving reranker cut)
        reranked_titles = {r.passage.title for r in reranked}
        rerank_recall = len(gold_titles & reranked_titles) / len(gold_titles) if gold_titles else 0.0
        diag["rerank"].append(rerank_recall)

        # Diagnostic: sentence recall (gold (title, sent_idx) in selected sentences)
        selected_facts = set()
        for r in reranked:
            for sent_str in r.supporting_sentences:
                # Find sentence index in passage
                for idx, s in enumerate(r.passage.sentences):
                    if " ".join(s.split()) == " ".join(sent_str.split()) or " ".join(s.split()).startswith(" ".join(sent_str.split())[:50]):
                        selected_facts.add((r.passage.title, idx))
                        break
        sentence_recall = len(gold_facts & selected_facts) / len(gold_facts) if gold_facts else 0.0
        diag["sentence"].append(sentence_recall)

        pb_output = pb.build(ex.question, reranked, use_citation_selection=True)
        try:
            gen_output = gen.generate(
                prompt=pb_output.prompt,
                target_model=pb_output.target_model,
                temperature=(
                    cfg.prompt_builder.temperature_large_model
                    if pb_output.target_model == "complex"
                    else cfg.prompt_builder.temperature_small_model
                ),
                supporting_fact_indices=pb_output.supporting_fact_indices,
                fact_mapping=pb_output.fact_mapping,
            )
            answer = gen_output.answer
            supporting_facts = gen_output.supporting_facts
            full_response = gen_output.full_response
        except Exception as e:
            log.warning(f"Generation failed for {ex.id}: {e}")
            answer = "Cannot answer based on the provided evidence."
            supporting_facts = []
            full_response = ""

        # Diagnostic: prediction recall (gold facts in LLM output)
        pred_facts = {(sf[0], sf[1]) for sf in supporting_facts} if supporting_facts else set()
        pred_recall = len(gold_facts & pred_facts) / len(gold_facts) if gold_facts else 0.0
        diag["prediction"].append(pred_recall)

        #dict indexed by ID, no "id" field needed
        predictions[ex.id] = {
            "question": ex.question,
            "gold_answer": ex.answer,
            "answer": answer,
            "sp": supporting_facts,  # List of [title, sentence_index]
            "full_response": full_response,
        }

    # Summary stats
    empty_sp = sum(1 for p in predictions.values() if not p["sp"])
    empty_ans = sum(1 for p in predictions.values() if not p["answer"] or p["answer"] == "Cannot answer based on the provided evidence.")
    log.info(f"Pipeline summary: {len(predictions)} predictions, {empty_sp} with empty SP, {empty_ans} with no/abstain answer")

    # Stage-by-stage recall diagnostics
    n = len(diag["retrieval"])
    if n > 0:
        log.info(f"\n{'='*60}")
        log.info(f"STAGE-BY-STAGE RECALL DIAGNOSTICS (n={n})")
        log.info(f"{'='*60}")
        for stage, values in diag.items():
            mean_val = sum(values) / len(values) if values else 0.0
            perfect = sum(1 for v in values if v >= 1.0)
            zero = sum(1 for v in values if v <= 0.0)
            log.info(f"  {stage:>12s}: mean={mean_val:.1%}  perfect={perfect}/{n}  zero={zero}/{n}")
        log.info(f"{'='*60}")

    return predictions


def save_predictions(predictions: Dict, output_path: Path):
    """Save predictions to JSON file (dict indexed by example ID)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)
    log.info(f"Saved {len(predictions)} predictions to {output_path}")


def evaluate_predictions(pred_path: Path, gold_path: Path, output_path: Optional[Path] = None):
    eval_script = Path("scripts/hotpot_evaluate_v1.py")

    if not eval_script.exists():
        log.error(f"Evaluation script not found at {eval_script}")
        return None

    log.info(f"Running official HotpotQA evaluation...")
    try:
        result = subprocess.run(
            ["python", str(eval_script), str(pred_path), str(gold_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0:
            log.info("Evaluation output:")
            print(result.stdout)

            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as f:
                    f.write(result.stdout)
                log.info(f"Saved evaluation results to {output_path}")
        else:
            log.error(f"Evaluation failed: {result.stderr}")

    except Exception as e:
        log.error(f"Failed to run evaluation: {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the full pipeline on HotpotQA")
    parser.add_argument(
        "--split",
        choices=["dev_distractor", "dev_fullwiki", "test_fullwiki"],
        default="dev_distractor",
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples (for quick testing)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/predictions.json"),
        help="Output path for predictions",
    )
    parser.add_argument(
        "--eval",
        type=Path,
        default=None,
        help="Path to gold answers JSON for evaluation",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Output path for evaluation metrics",
    )
    args = parser.parse_args()

    cfg = load_config()

    if args.split == "dev_fullwiki":
        cfg.data.dev_distractor = cfg.data.dev_fullwiki
    elif args.split == "test_fullwiki":
        cfg.data.dev_distractor = cfg.data.test_fullwiki

    if args.limit:
        cfg.eval.limit = args.limit

    log.info(f"{'='*80}")
    log.info(f"Evaluating on: {args.split}")
    if args.limit:
        log.info(f"Limit: {args.limit} examples")
    log.info(f"Output: {args.output}")
    log.info(f"{'='*80}")

    examples, retriever, reranker, pb, gen, cfg = build_pipeline(cfg)

    predictions = run_pipeline(examples, retriever, reranker, pb, gen, cfg, limit=args.limit)

    save_predictions(predictions, args.output)

    if args.eval:
        if not args.eval.exists():
            log.error(f"Gold file not found: {args.eval}")
        else:
            evaluate_predictions(args.output, args.eval, args.metrics)

    log.info(f"{'='*80}")
    log.info("Done!")

if __name__ == "__main__":
    main()
