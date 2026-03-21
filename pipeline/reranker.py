import os
import re
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from sentence_transformers import CrossEncoder

from pipeline.data_loader import Passage
from pipeline.embedder import OllamaEmbedder
from pipeline.indexer import RetrievalResult
from scripts.logger import get_logger

log = get_logger("reranker")


@dataclass
class RerankResult:
    passage: Passage
    score: float                              
    rank: int                                 
    retrieval_rank: int                       
    retrieval_score: float                    
    supporting_sentences: List[str] = field(default_factory=list)
    supporting_sentence_indices: List[int] = field(default_factory=list)
    sentence_scores: List[float] = field(default_factory=list)
    hop: int = 0

class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        sentence_model_name: str = "nomic-embed-text",
        ollama_base_url: str = "http://localhost:11434",
        device: str = "cpu",
        sentence_score_threshold: float = 0.4,
        max_sentences_per_passage: int = 3,
        batch_size: int = 32,
        sentence_passage_limit: int = 3,
        title_overlap_boost: float = 0.05,
    ):
        self.model_name = model_name
        self.sentence_model_name = sentence_model_name
        self.device = device
        self.sentence_score_threshold = sentence_score_threshold
        self.max_sentences_per_passage = max_sentences_per_passage
        self.batch_size = batch_size
        self.sentence_passage_limit = sentence_passage_limit
        self.title_overlap_boost = title_overlap_boost

        log.info(f"Loading cross-encoder: {model_name}")
        self._cross_encoder = CrossEncoder(model_name, device=device)

        log.info(f"Loading sentence encoder via Ollama: {sentence_model_name}")
        self._sentence_encoder = OllamaEmbedder(
            model=sentence_model_name,
            base_url=ollama_base_url,
            batch_size=batch_size,
        )

        log.success("Reranker ready.")

    @classmethod
    def from_config(cls, cfg=None) -> "Reranker":
        from scripts.config import load_config
        if cfg is None or isinstance(cfg, (str,)):
            cfg = load_config(cfg)
        r = cfg.reranker
        ollama_url = cfg.generator.ollama_base_url  
        return cls(
            model_name=r.model_name,
            sentence_model_name=r.sentence_model_name,
            ollama_base_url=ollama_url,
            device=r.device,
            sentence_score_threshold=r.sentence_score_threshold,
            max_sentences_per_passage=r.max_sentences_per_passage,
            batch_size=r.batch_size,
            sentence_passage_limit=getattr(r, "sentence_passage_limit", 3),
            title_overlap_boost=getattr(r, "title_overlap_boost", 0.05),
        )

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: Optional[int] = None,
        select_sentences: bool = True,
    ) -> List[RerankResult]:
        if not candidates:
            return []
        
        pairs = [(query, c.passage.title_text) for c in candidates]

        log.step(f"Cross-encoder scoring {len(pairs)} passages...")
        scores: np.ndarray = self._cross_encoder.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )

        results = [
            RerankResult(
                passage=cand.passage,
                score=float(score),
                rank=0,
                retrieval_rank=cand.rank,
                retrieval_score=cand.score,
                hop=cand.hop,
            )
            for cand, score in zip(candidates, scores)
        ]

        results.sort(key=lambda r: r.score, reverse=True)

        if top_k is not None:
            results = results[:top_k]

        for i, r in enumerate(results):
            r.rank = i

        if select_sentences:
            self._select_supporting_sentences(query, results)

        return results

    def _normalize_tokens(self, text: str) -> set:
        """Normalize text to lowercase alphanumeric tokens for robust matching."""
        return {
            re.sub(r"[^a-z0-9]+", "", t.lower())
            for t in text.split()
            if re.sub(r"[^a-z0-9]+", "", t.lower())
        }

    def _select_supporting_sentences(
        self, query: str, results: List[RerankResult]
    ) -> None:
        # Only select sentences from top N reranked passages (saves embedding calls)
        active_results = results[:self.sentence_passage_limit]

        passage_sentence_ranges: List[Tuple[int, int]] = []
        all_sentences: List[str] = []

        # Track mapping from flat index back to original sentence index in passage
        orig_indices_map: List[List[int]] = []  # per-passage: list of original indices for non-empty sentences

        for r in active_results:
            start = len(all_sentences)
            orig_indices = []
            for orig_idx, s in enumerate(r.passage.sentences):
                stripped = s.strip()
                if stripped:
                    all_sentences.append(stripped)
                    orig_indices.append(orig_idx)
            passage_sentence_ranges.append((start, start + len(orig_indices)))
            orig_indices_map.append(orig_indices)

        if not all_sentences:
            for r in results:
                r.supporting_sentence_indices = []
                r.supporting_sentences = []
                r.sentence_scores = []
            return

        query_vec: np.ndarray = self._sentence_encoder.encode_query(query)
        sent_vecs: np.ndarray = self._sentence_encoder.encode(all_sentences)

        # Cosine similarity = dot product on unit vectors
        sim_scores: np.ndarray = sent_vecs @ query_vec  # shape: (n_sentences,)

        query_words = self._normalize_tokens(query)
        for r_idx, (r, (start, end), orig_indices) in enumerate(zip(active_results, passage_sentence_ranges, orig_indices_map)):
            if end <= start:
                continue

            sents = all_sentences[start:end]
            scores = sim_scores[start:end].tolist()

            # Title-match boost: if passage title words overlap with query, boost all sentence scores
            title_words = self._normalize_tokens(r.passage.title)
            if title_words & query_words:
                scores = [s + self.title_overlap_boost for s in scores]

            # Zip scores, sentences, and original indices together, sort by score descending
            paired = sorted(zip(scores, sents, orig_indices), reverse=True)
            selected: List[Tuple[float, str, int]] = []

            # Rank-aware guaranteed minimum: top passages get more sentences
            if r.rank <= 1:
                min_guarantee = 4  # Top 2 passages: guarantee 4 sentences
            elif r.rank <= 3:
                min_guarantee = 3  # Middle passages: guarantee 3
            else:
                min_guarantee = 2  # Lower passages: guarantee 2

            for i, (score, sent, orig_idx) in enumerate(paired):
                if i < min_guarantee:
                    selected.append((score, sent, orig_idx))
                elif score >= self.sentence_score_threshold and len(selected) < self.max_sentences_per_passage:
                    selected.append((score, sent, orig_idx))
                else:
                    break  # Sorted descending — all remaining are below threshold

            # Sort selected sentences chronologically (by original index) for coherent reading
            selected.sort(key=lambda x: x[2])

            r.supporting_sentences = [s[1] for s in selected]
            r.supporting_sentence_indices = [s[2] for s in selected]
            r.sentence_scores = [round(float(s[0]), 4) for s in selected]

        # Clear sentence fields for lower-ranked passages we skipped
        for r in results[self.sentence_passage_limit:]:
            r.supporting_sentence_indices = []
            r.supporting_sentences = []
            r.sentence_scores = []

    def __repr__(self) -> str:
        return (
            f"Reranker(model={self.model_name}, "
            f"threshold={self.sentence_score_threshold}, "
            f"max_sents={self.max_sentences_per_passage}, "
            f"device={self.device})"
        )


if __name__ == "__main__":
    from pipeline.data_loader import HotpotQALoader
    from pipeline.indexer import HybridRetriever
    from scripts.config import load_config

    cfg = load_config()
    loader = HotpotQALoader(cfg.data.dev_distractor)
    examples = loader.load(limit=50)

    seen = set()
    passages = []
    for ex in examples:
        for ctx in ex.contexts:
            key = (ctx.title, ctx.text)
            if key not in seen:
                seen.add(key)
                passages.append(ctx)

    retriever = HybridRetriever.from_config(cfg)
    retriever.index(passages, show_progress=False)
    reranker = Reranker.from_config(cfg)

    ex = examples[0]
    log.info(f"Q:    {ex.question}")
    log.info(f"A:    {ex.answer}")
    log.info(f"Gold: {[sf.title for sf in ex.supporting_facts]}")

    retrieved = retriever.retrieve_multihop(
        ex.question, hops=2, top_k=20, question_type=ex.question_type
    )
    log.info(f"Retrieved {len(retrieved)} candidates")

    reranked = reranker.rerank(ex.question, retrieved, top_k=5)

    log.info("--- Re-ranked results ---")
    for r in reranked:
        log.info(
            f"[{r.rank}] score={r.score:.3f}  (was retrieval #{r.retrieval_rank})  | {r.passage.title}"
        )
        for sent, sc in zip(r.supporting_sentences, r.sentence_scores):
            log.info(f"      [{sc:.3f}] {sent[:120]}")
