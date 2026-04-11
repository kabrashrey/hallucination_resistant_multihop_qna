"""
Unified Pipeline Test Script
Extracts the redundant __main__ boilerplate from the pipeline modules into a single fast test.
"""

from scripts.config import load_config
from scripts.logger import get_logger
from pipeline.data_loader import HotpotQALoader
from pipeline.indexer import HybridRetriever
from pipeline.reranker import Reranker
from pipeline.prompt_builder import PromptBuilder
from pipeline.generator import Generator
from pipeline.verifier import Verifier
from pipeline.decider import Decider

log = get_logger("test_pipeline")

def test_single_example():
    cfg = load_config()
    
    # Load data
    loader = HotpotQALoader(cfg.data.dev_distractor)
    examples = loader.load(limit=10)
    
    seen = set()
    passages = []
    for ex in examples:
        for ctx in ex.contexts:
            key = (ctx.title, ctx.text)
            if key not in seen:
                seen.add(key)
                passages.append(ctx)
                
    # Initialize pipeline
    log.info(f"Building indexer with {len(passages)} passages...")
    retriever = HybridRetriever.from_config(cfg)
    retriever.index(passages, show_progress=False)
    
    reranker = Reranker.from_config(cfg)
    pb = PromptBuilder.from_config(cfg)
    gen = Generator.from_config(cfg)
    verifier = Verifier.from_config(cfg)
    decider = Decider()
    
    ex = examples[0]
    log.info(f"\n{'='*80}")
    log.info(f"Example: {ex.id}")
    log.info(f"Question: {ex.question}")
    log.info(f"Gold answer: {ex.answer}")
    
    # 1. Retrieve
    base_top_k = cfg.retriever.top_k
    base_top_k_per_hop = cfg.retriever.multihop.top_k_per_hop

    def run_attempt(top_k: int, top_k_per_hop: int):
        retrieved = retriever.retrieve_multihop(
            ex.question,
            hops=cfg.retriever.multihop.hops,
            top_k=top_k,
            top_k_per_hop=top_k_per_hop,
            question_type=ex.question_type,
        )
        log.info(f"Retrieved {len(retrieved)} candidates")

        reranked = reranker.rerank(ex.question, retrieved, top_k=cfg.reranker.top_k)
        log.info(f"Re-ranked to {len(reranked)} passages")

        pb_output = pb.build(ex.question, reranked)
        log.info(f"Complexity score: {pb_output.complexity_score:.3f}")
        log.info(f"Target model: {pb_output.target_model}")

        temperature = (
            cfg.prompt_builder.temperature_large_model
            if pb_output.target_model == "complex"
            else cfg.prompt_builder.temperature_small_model
        )
        gen_output = gen.generate(
            prompt=pb_output.prompt,
            target_model=pb_output.target_model,
            temperature=temperature,
            supporting_fact_indices=pb_output.supporting_fact_indices,
            fact_mapping=pb_output.fact_mapping
        )
        return reranked, gen_output

    reranked, gen_output = run_attempt(base_top_k, base_top_k_per_hop)
    
    # 5. Verify 
    verify_result = verifier.verify(
        answer=gen_output.answer,
        evidence=reranked,
        supporting_facts=gen_output.supporting_facts,
    )
    def retry_fn():
        log.info("Verifier rejected; retrying with expanded retrieval.")
        retry_reranked, retry_gen_output = run_attempt(base_top_k * 2, base_top_k_per_hop * 2)
        return {
            "answer": retry_gen_output.answer,
            "supporting_facts": retry_gen_output.supporting_facts,
            "reranked_results": retry_reranked,
            "full_response": retry_gen_output.full_response,
        }

    decision, attempt = decider.decide(
        answer=gen_output.answer,
        verification=verify_result,
        reranked_results=reranked,
        supporting_facts=gen_output.supporting_facts,
        verifier=verifier,
        retry_fn=retry_fn,
    )
    log.info(f"\n{'='*80}")
    log.info(f"Generated answer: {attempt['answer']}")
    log.info(f"Supporting facts: {attempt['supporting_facts']}")
    log.info(f"Model: {gen_output.model_used}")
    log.info(f"Generation time: {gen_output.generation_time:.2f}s")
    log.info(f"Support score: {decision.support_score:.4f}")
    log.info(f"Is supported: {decision.verifier_supported}")
    log.info(f"Final answer: {decision.answer}")
    log.info(f"Supporting passage ids: {decision.supporting_passage_ids}")
    log.info(f"Confidence: {decision.confidence:.4f}")
    log.info(f"{'='*80}\n")

if __name__ == "__main__":
    test_single_example()
