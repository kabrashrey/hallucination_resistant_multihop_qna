"""
Hybrid Multi-Hop Retriever: BM25 + FAISS
The hybrid approach covers both cases — BM25 catches exact term matches that dense might miss, 
and dense catches semantic similarity that BM25 can't understand.
"""

import os
# Prevent FAISS segfaults on macOS must be set before importing FAISS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import pickle
import numpy as np
from pathlib import Path
import faiss
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
from pipeline.embedder import OllamaEmbedder
from pipeline.data_loader import Passage
from scripts.config import load_config
from scripts.logger import get_logger
log = get_logger("indexer")


@dataclass
class RetrievalResult:
    """
    Eg:
        passage: Passage(title="Scott Derrickson", sentences=[...])
        score: 0.87
        rank: 0
        bm25_score: 0.45
        dense_score: 0.92
        confidence: 0.73
    """
    passage: Passage     # from data_loader
    score: float         # final combined score
    rank: int            # rank of the passage (0-indexed)
    bm25_score: float    # raw BM25 score (before normalization)
    dense_score: float   # raw cosine similarity from FAISS (-1 to 1)
    confidence: float = 0.0  # how confident we are (score gap from #1 to #2)
    hop: int = 0         # which hop retrieved this passage (0 = first hop)
    # Example: score=0.929 (bm25=17.59, dense=0.541) --> high overlap + high semantic similarity


def _tokenize(text: str) -> List[str]:
    """Lowercase and split texts by whitespaces for BM25"""
    return text.lower().split()

def _rrf_score(rank: int, k: int = 60) -> float:
    """
    BM25 scores might be 0 to 25 while dense scores are -1 to 1, not comparable. RRF depends only on rank
    Reciprocal Rank Fusion score for a single rank position
    If a document is rank 0 then RRF score is 1/61, rank 1 then 1/62, rank 2 then 1/63, and so on

    Low Rank --> high RRF score

    score = 1 / (k + rank + 1)
    """
    return 1.0 / (k + rank + 1)


def _compute_confidence(scores: List[float]) -> float:
    """
    Confidence signal: how much better is top result #1 than #2?
    Returns 0-1: high = top result clearly dominates, low = top results are close (ambiguous)
    """
    if len(scores) < 2 or scores[0] <= 0:
        return 0.0
    gap = (scores[0] - scores[1]) / scores[0]
    return min(max(gap, 0.0), 1.0)


class BM25Retriever:
    """
    Sparse retriever using BM25 (Okapi)
    Scores passages by term-frequency overlap with the query

    How many times does the query appear in the passage? - TF
    How important is the query term in the corpus? -  IDF
    How long is the passage? - Longer passage get penalized
    """

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_tokens: List[List[str]] = []

    def index(self, texts: List[str]):
        """
        Build BM25 index from passage texts, stores token frequency and IDF values
        texts: list of passage strings (title + text)
        """
        self._corpus_tokens = [_tokenize(t) for t in texts] # tokenize passage
        self._bm25 = BM25Okapi(self._corpus_tokens) # build BM25 inverted index

    # def score(self, query: str) -> np.ndarray:
    #     """
    #     Score all passages for a query
    #     Returns array of shape (n_passages,), higher = more relevant
    #     WARNING: scores entire corpus — use top_k() for FullWiki scale
    #     """
    #     if self._bm25 is None:
    #         raise RuntimeError("Call .index() first")
    #     return self._bm25.get_scores(_tokenize(query)) # get scores for all passages

    def top_k(self, query: str, k: int) -> List[Tuple[int, float]]:
        """
        Get top-k passage indices
        Returns scores without fully ranking the corpus
        Returns: list of (passage_index, bm25_score) sorted descending
        """
        if self._bm25 is None:
            raise RuntimeError("Call .index() first")
        scores = self._bm25.get_scores(_tokenize(query))
        
        # partial sort:
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def save(self, path: Path):
        """Save BM25 corpus tokens to disk"""
        with open(path / "bm25_tokens.pkl", "wb") as f:
            pickle.dump(self._corpus_tokens, f)

    def load(self, path: Path):
        """Load BM25 corpus tokens from disk and rebuild index"""
        with open(path / "bm25_tokens.pkl", "rb") as f:
            self._corpus_tokens = pickle.load(f)
        self._bm25 = BM25Okapi(self._corpus_tokens)

    @property
    def is_indexed(self) -> bool:
        return self._bm25 is not None


class DenseRetriever:
    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        ollama_base_url: str = "http://localhost:11434",
        batch_size: int = 64,
        # legacy param kept so from_config() call sites don't break
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.encoder = OllamaEmbedder(
            model=model_name,
            base_url=ollama_base_url,
            batch_size=batch_size,
        )
        self._index: Optional[faiss.IndexFlatIP] = None
        self._dim: int = self.encoder.dim

    def index(self, texts: List[str], show_progress: bool = True):
        vectors = self.encoder.encode(texts, show_progress=show_progress)
        self._dim = vectors.shape[1]

        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(vectors)

    def score(self, query: str) -> np.ndarray:
        if self._index is None:
            raise RuntimeError("Call .index() first")

        q_vec = self.encoder.encode_query(query).reshape(1, -1)
        n = self._index.ntotal
        distances, indices = self._index.search(q_vec, n)

        scores = np.zeros(n, dtype=np.float32)
        scores[indices[0]] = distances[0]
        return scores

    def search_top_k(self, query: str, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._index is None:
            raise RuntimeError("Call .index() first")

        q_vec = self.encoder.encode_query(query).reshape(1, -1)
        distances, indices = self._index.search(q_vec, top_k)
        return distances[0], indices[0]

    def encode_query(self, query: str) -> np.ndarray:
        return self.encoder.encode_query(query)

    def save(self, path: Path):
        faiss.write_index(self._index, str(path / "faiss.index"))
        with open(path / "dense_meta.json", "w") as f:
            json.dump({"dim": self._dim, "model_name": self.model_name}, f)

    def load(self, path: Path):
        self._index = faiss.read_index(str(path / "faiss.index"))
        with open(path / "dense_meta.json", "r") as f:
            meta = json.load(f)
        self._dim = meta["dim"]

    @property
    def is_indexed(self) -> bool:
        return self._index is not None


class HybridRetriever:
    def __init__(
        self,
        embed_model: str = "nomic-embed-text",
        ollama_base_url: str = "http://localhost:11434",
        extraction_model: str = "qwen2.5:7b",
        alpha: float = 0.7,
        alpha_bridge: Optional[float] = None,
        alpha_comparison: Optional[float] = None,
        rrf_k: int = 20,
        candidate_pool_size: int = 100,
        device: str = "cpu",   # kept for API compat, unused (Ollama handles device)
        batch_size: int = 64,
        prompts=None,
        max_bridge_entities: int = 3,
        extraction_temperature: float = 0.0,
        extraction_timeout: int = 120,
    ):
        self.extraction_model = extraction_model
        self.ollama_base_url = ollama_base_url
        self.alpha = alpha
        self.alpha_bridge = alpha_bridge
        self.alpha_comparison = alpha_comparison
        self.rrf_k = rrf_k
        self.candidate_pool_size = candidate_pool_size
        self.prompts = prompts
        self.max_bridge_entities = max_bridge_entities
        self.extraction_temperature = extraction_temperature
        self.extraction_timeout = extraction_timeout

        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(
            model_name=embed_model,
            ollama_base_url=ollama_base_url,
            batch_size=batch_size,
        )
        self._passages: List[Passage] = []
        self._texts: List[str] = []

    @classmethod
    def from_config(cls, cfg=None) -> "HybridRetriever":
        """
        Config Retriever
        Eg: retriever = HybridRetriever.from_config("configs/fast.yaml") # custom config
        """
        if cfg is None or isinstance(cfg, (str, Path)):
            cfg = load_config(cfg)
        r = cfg.retriever
        return cls(
            embed_model=r.embed_model,
            ollama_base_url=r.ollama_base_url,
            extraction_model=cfg.generator.model_small,
            alpha=r.alpha,
            alpha_bridge=r.alpha_bridge,
            alpha_comparison=r.alpha_comparison,
            rrf_k=r.rrf_k,
            candidate_pool_size=r.candidate_pool_size,
            device=r.device,
            batch_size=r.batch_size,
            prompts=getattr(cfg, "prompts", None),
            max_bridge_entities=r.multihop.max_bridge_entities,
            extraction_temperature=getattr(r.multihop, "extraction_temperature", 0.0),
            extraction_timeout=getattr(r.multihop, "extraction_timeout", 120),
        )

    def _get_alpha(self, question_type: Optional[str] = None) -> float:
        if question_type == "bridge" and self.alpha_bridge is not None:
            return self.alpha_bridge
        if question_type == "comparison" and self.alpha_comparison is not None:
            return self.alpha_comparison
        return self.alpha

    def index(self, passages: Union[List[Passage], List[str]], show_progress: bool = True):
        """
        Build both BM25 and FAISS indices

        passages: either list of Passage objects (from data_loader)
                  or list of plain strings
        """
        if len(passages) == 0:
            raise ValueError("No passages to index")

        # Accept either Passage objects or raw strings
        if isinstance(passages[0], Passage):
            self._passages = list(passages)
            self._texts = [p.title_text for p in self._passages]
        else:
            self._passages = [Passage(title="", sentences=[str(p)]) for p in passages]
            self._texts = [str(p) for p in passages]

        log.info(f"Indexing {len(self._texts)} passages...")

        # Build both indices
        log.step("Building BM25 index...")
        self.bm25.index(self._texts)

        log.step("Building FAISS dense index...")
        self.dense.index(self._texts, show_progress=show_progress)

        log.success(f"{len(self._texts)} passages indexed.")

    def save(self, save_dir: Union[str, Path]):
        """
        Save both indices + passage data
        call .load() instead of re-encoding
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        self.dense.save(save_dir)

        # Save BM25 tokens
        self.bm25.save(save_dir)

        # Save passage data
        passages_data = []
        for p in self._passages:
            passages_data.append({
                "title": p.title,
                "sentences": p.sentences,
                "passage_id": p.passage_id,
            })
        with open(save_dir / "passages.json", "w", encoding="utf-8") as f:
            json.dump(passages_data, f, ensure_ascii=False)

        # Save config
        config = {
            "alpha": self.alpha,
            "alpha_bridge": self.alpha_bridge,
            "alpha_comparison": self.alpha_comparison,
            "rrf_k": self.rrf_k,
            "num_passages": len(self._passages),
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        log.success(f"Saved index to {save_dir}/ ({len(self._passages)} passages)")

    def load(self, save_dir: Union[str, Path]):
        """
        Load indices + passages from disk (skips encoding)
        The SentenceTransformer model is still loaded for query encoding
        """
        save_dir = Path(save_dir)
        if not (save_dir / "faiss.index").exists():
            raise FileNotFoundError(f"No saved index in {save_dir}")

        log.info(f"Loading index from {save_dir}/...")

        # Load FAISS
        self.dense.load(save_dir)

        # Load BM25
        self.bm25.load(save_dir)

        # Load passages
        with open(save_dir / "passages.json", "r", encoding="utf-8") as f:
            passages_data = json.load(f)
        self._passages = [
            Passage(title=p["title"], sentences=p["sentences"], passage_id=p.get("passage_id"))
            for p in passages_data
        ]
        self._texts = [p.title_text for p in self._passages]

        # Load config
        with open(save_dir / "config.json", "r") as f:
            config = json.load(f)
        self.alpha = config.get("alpha", self.alpha)
        self.alpha_bridge = config.get("alpha_bridge", self.alpha_bridge)
        self.alpha_comparison = config.get("alpha_comparison", self.alpha_comparison)
        self.rrf_k = config.get("rrf_k", self.rrf_k)

        log.success(f"Loaded index from {save_dir}/ ({len(self._passages)} passages)")

    def retrieve(self, query: str, top_k: int = 10,
                 question_type: Optional[str] = None,
                 candidate_pool_size: Optional[int] = None) -> List[RetrievalResult]:
        """
        Retrieve top-k passages using candidate-pool RRF scoring

          1. Get top-N candidates from BM25 (fast, uses argpartition)
          2. Get top-N candidates from FAISS (fast, native top-k)
          3. Union the candidate pools (~2N passages at most)
          4. RRF rank only within this small pool

        query: the question string
        top_k: number of passages to return
        question_type: "bridge" or "comparison" (uses per-type alpha if set)
        candidate_pool_size: how many candidates to pull from each retriever before fusing

        Returns: sorted list of RetrievalResult (highest score first)
        """
        if not self.bm25.is_indexed or not self.dense.is_indexed:
            raise RuntimeError("Call .index() first")

        alpha = self._get_alpha(question_type)
        pool_size = min(candidate_pool_size or self.candidate_pool_size, len(self._passages))

        # Get top-N candidates from each retriever (fast)
        bm25_top = self.bm25.top_k(query, pool_size)          # [(idx, score), ...]
        dense_scores_arr, dense_indices_arr = self.dense.search_top_k(query, pool_size)  # (scores, indices)

        # Build candidate pool (union of both top-N sets)
        candidate_indices = set()
        bm25_score_map = {}   # idx -> raw bm25 score
        dense_score_map = {}  # idx -> raw dense score

        for rank, (idx, score) in enumerate(bm25_top):
            candidate_indices.add(idx)
            bm25_score_map[idx] = (rank, score)   # (rank_in_bm25, raw_score)

        for rank, (idx, score) in enumerate(zip(dense_indices_arr.tolist(), dense_scores_arr.tolist())):
            candidate_indices.add(idx)
            dense_score_map[idx] = (rank, score)

        # RRF score only the candidates
        scored = []
        for idx in candidate_indices:
            # If a passage was found by one retriever but not other, assign it a worst-case rank (pool_size) for the missing retriever
            bm25_rank = bm25_score_map[idx][0] if idx in bm25_score_map else pool_size
            dense_rank = dense_score_map[idx][0] if idx in dense_score_map else pool_size
            bm25_raw = bm25_score_map[idx][1] if idx in bm25_score_map else 0.0
            dense_raw = dense_score_map[idx][1] if idx in dense_score_map else 0.0

            rrf = alpha * _rrf_score(dense_rank, self.rrf_k) + \
                  (1 - alpha) * _rrf_score(bm25_rank, self.rrf_k)

            scored.append((idx, rrf, bm25_raw, dense_raw))

        # Sort by RRF score and take top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top_k]

        # Compute confidence: how much #1 dominates over #2
        all_scores = [s[1] for s in scored]
        conf = _compute_confidence(all_scores)

        results = []
        for rank, (idx, rrf, bm25_raw, dense_raw) in enumerate(scored):
            results.append(RetrievalResult(
                passage=self._passages[idx],
                score=float(rrf),
                rank=rank,
                bm25_score=float(bm25_raw),
                dense_score=float(dense_raw),
                confidence=conf if rank == 0 else 0.0,  # confidence only on top result
                hop=0,
            ))
        return results

    @staticmethod
    def _extract_bridge_entities(passage: Passage, question: str) -> List[str]:
        """
        Extract potential bridge entities from a passage that could be used for hop 2

        Catches:
          - Passage title (primary source for entity names)
          - Multi-word names:   "Shirley Temple", "New York City"
          - Acronyms:           "NASA", "FBI", "UNESCO"
          - Single-name entities: "Cher", "Madonna", "Beyoncé"
          - Titles with punctuation: "Dr. Strange", "St. Louis"
          - Parenthetical context: "(1928-1992)", "(actress)"
        """
        import re
        question_lower = question.lower()
        text = passage.text

        entities = []

        # Passage title is primary bridge source (often contains entity names like "Shirley Temple")
        if passage.title and passage.title.strip():
            entities.append(passage.title)

        # Multi-word capitalized phrases ("Shirley Temple", "New York City", "Dr. Strange", "St. Louis")
        # Improved pattern: catches 2+ consecutive capitalized words
        entities += re.findall(r'\b([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)

        # Acronyms — 2+ consecutive uppercase letters ("NASA", "FBI", "MCU")
        entities += re.findall(r'\b([A-Z]{2,})\b', text)

        # Single capitalized words that appear mid-sentence, exclude sentence starters by requiring a lowercase letter or punctuation before
        entities += re.findall(r'(?<=[a-z.,;:!?]\s)([A-Z][a-zà-ÿ]{2,})\b', text)

        # Parenthetical context (dates, descriptions) - "Shirley Temple (1928-1992)"
        # Extract text in parentheses that might be useful context
        entities += re.findall(r'\(([^)]+)\)', text)[:3]  # limit to 3 parenthetical contexts

        # Filter out entities already in the question
        bridge = []
        seen = set()
        for entity in entities:
            entity_clean = entity.strip().rstrip('.')
            if (entity_clean.lower() not in question_lower
                    and entity_clean not in seen
                    and len(entity_clean) > 1
                    and entity_clean != passage.title):  # don't duplicate title
                seen.add(entity_clean)
                bridge.append(entity_clean)

        # Always include passage title as fallback if not already added
        if passage.title and passage.title.strip() and passage.title not in bridge:
            bridge.insert(0, passage.title)

        return bridge[:self.max_bridge_entities]

    def _extract_entities_llm(self, passages: List[Passage], question: str) -> List[str]:
        """Use local LLM via Ollama to extract bridge entities from passages."""
        if not passages:
            return []
        import requests
        import re

        system_prompt = self.prompts.indexer_system.strip() if self.prompts and self.prompts.indexer_system else (
            "You are a precise entity extractor. Respond ONLY with a valid JSON array of strings, with no additional text and no markdown formatting (do not use ```json)."
        )

        context = ""
        for i, p in enumerate(passages):
            # Truncate text to 800 chars to save prompt context and speed up extraction
            context += f"Passage {i+1} Title: {p.title}\nPassage {i+1} Text: {p.text[:800]}\n\n"

        if self.prompts and self.prompts.indexer_user:
            prompt = self.prompts.indexer_user.format(question=question, context=context)
        else:
            prompt = (
                f"Question: {question}\n\n"
                f"Context:\n{context}\n"
                f"Extract the most important named entities (people, places, organizations, specific concepts) from the context that are relevant to finding the final answer to the question. "
                f"Focus particularly on entities that link different pieces of information together (bridge entities).\n"
                f"DO NOT include entities that are already explicitly mentioned in the question.\n"
                f"Output strictly a JSON list of strings, for example: [\"Entity 1\", \"Entity 2\"]. Output nothing else."
            )

        try:
            url = f"{self.ollama_base_url}/api/generate"
            payload = {
                "model": self.extraction_model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "temperature": self.extraction_temperature,
                "options": {"num_ctx": 4096},
                "keep_alive": -1,
            }
            resp = requests.post(url, json=payload, timeout=self.extraction_timeout)
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()

            entities = []
            try:
                entities = json.loads(text)
            except json.JSONDecodeError:
                match = re.search(r'\[.*?\]', text, re.DOTALL)
                if match:
                    try:
                        entities = json.loads(match.group(0))
                    except Exception:
                        pass

            if isinstance(entities, list):
                # Also collect passage titles as robust fallbacks
                titles = [p.title for p in passages if p.title]
                entities.extend(titles)
                # Deduplicate and filter out question terms
                q_lower = question.lower()
                final_entities = []
                seen = set()
                for e in entities:
                    e_clean = str(e).strip()
                    if e_clean and e_clean.lower() not in q_lower and e_clean.lower() not in seen:
                        seen.add(e_clean.lower())
                        final_entities.append(e_clean)
                return final_entities
        except Exception as e:
            log.warning(f"LLM entity extraction failed: {e}. Falling back to regex.")

        # Fallback to regex
        fallback = []
        for p in passages:
            fallback.extend(self._extract_bridge_entities(p, question))
        return list(dict.fromkeys(fallback))

    def retrieve_multihop(self, query: str, hops: int = 2, top_k: int = 10,
                          top_k_per_hop: int = 5,
                          question_type: Optional[str] = None) -> List[RetrievalResult]:
        """
        Multi-hop iterative retrieval for bridge questions.

        Hop 1: retrieve using the original question
        Hop 2: extract named entities from hop 1 passages via LLM,
               reformulate query = question + bridge entities, retrieve again
        Finally: merge, deduplicate, re-rank results from all hops

        query: the original question
        hops: number of retrieval iterations (default 2)
        top_k: total results to return after merging
        top_k_per_hop: passages retrieved per hop
        question_type: "bridge" or "comparison"

        Returns: deduplicated, re-ranked RetrievalResult list
        """
        all_results: List[RetrievalResult] = []
        seen_keys: set = set()
        current_query = query

        for hop in range(hops):
            hop_results = self.retrieve(current_query, top_k=top_k_per_hop,
                                        question_type=question_type)

            # Compute confidence for this hop
            hop_scores = [r.score for r in hop_results]
            hop_confidence = _compute_confidence(hop_scores)

            # Collect new (unseen) passages
            new_in_hop = 0
            top_passages_for_extraction = []
            for r in hop_results[:5]:
                key = (r.passage.title, r.passage.text)
                if key not in seen_keys:
                    seen_keys.add(key)
                    r.hop = hop
                    r.confidence = hop_confidence if new_in_hop == 0 else 0.0
                    new_in_hop += 1
                    all_results.append(r)
                top_passages_for_extraction.append(r.passage)

            # Extract bridge entities using LLM
            bridge_entities = []
            if hop < hops - 1 and top_passages_for_extraction:
                bridge_entities = self._extract_entities_llm(top_passages_for_extraction, query)
                log.info(f"Hop {hop} extracted entities: {bridge_entities}")

            # Reformulate query for next hop using extracted entities
            if hop < hops - 1 and bridge_entities:
                hop_passage_titles = {r.passage.title for r in hop_results}
                body_entities = [e for e in bridge_entities if e not in hop_passage_titles]
                title_entities = [e for e in bridge_entities if e in hop_passage_titles]
                unique_entities = list(dict.fromkeys(title_entities + body_entities))[:self.max_bridge_entities]
                entity_str = " ".join(unique_entities)
                current_query = f"{query} {entity_str}"
                log.info(f"Hop {hop+1} formulated query: {current_query}")

        # Re-rank all collected results by score (descending)
        all_results.sort(key=lambda r: r.score, reverse=True)

        # Re-assign ranks
        for i, r in enumerate(all_results):
            r.rank = i

        return all_results[:top_k]

    def retrieve_passages(self, query: str, top_k: int = 10,
                          question_type: Optional[str] = None) -> List[Passage]:
        """return just the Passage objects"""
        return [r.passage for r in self.retrieve(query, top_k, question_type)]

    def retrieve_texts(self, query: str, top_k: int = 10,
                       question_type: Optional[str] = None) -> List[str]:
        """return just the passage text strings"""
        return [r.passage.title_text for r in self.retrieve(query, top_k, question_type)]

    @property
    def num_passages(self) -> int:
        return len(self._passages)

    def __repr__(self) -> str:
        return (f"HybridRetriever(alpha={self.alpha}, "
                f"alpha_bridge={self.alpha_bridge}, "
                f"alpha_comparison={self.alpha_comparison}, "
                f"rrf_k={self.rrf_k}, "
                f"indexed={self.bm25.is_indexed}, "
                f"passages={len(self._passages)})")


if __name__ == "__main__":
    from pipeline.data_loader import HotpotQALoader
    cfg = load_config()

    # Load data using config
    loader = HotpotQALoader(cfg.data.dev_distractor)
    examples = loader.load(limit=50)

    # Collect unique passages
    seen = set()
    passages: List[Passage] = []
    for ex in examples:
        for ctx in ex.contexts:
            key = (ctx.title, ctx.text)
            if key not in seen:
                seen.add(key)
                passages.append(ctx)

    log.info(f"{len(passages)} unique passages from {len(examples)} examples")

    # Build retriever from config
    retriever = HybridRetriever.from_config(cfg)
    retriever.index(passages)

    # Save to disk
    retriever.save(cfg.retriever.index_cache_dir)

    # Load from disk
    retriever2 = HybridRetriever.from_config(cfg)
    retriever2.load(cfg.retriever.index_cache_dir)

    # Test single-hop (comparison)
    q1 = "Were Scott Derrickson and Ed Wood of the same nationality?"
    log.info(f"Q (comparison): {q1}")
    results = retriever2.retrieve(q1, top_k=cfg.retriever.top_k, question_type="comparison")
    for r in results[:3]:
        conf_str = f" conf={r.confidence:.2f}" if r.confidence > 0 else ""
        log.info(f"[{r.rank}] score={r.score:.4f}{conf_str} title={r.passage.title}")

    # Test multi-hop (bridge)
    q2 = "What government position was held by the woman who portrayed Corliss Archer?"
    log.info(f"Q (bridge, multi-hop): {q2}")
    mh = cfg.retriever.multihop
    results = retriever2.retrieve_multihop(q2, hops=mh.hops, top_k=5,
                                           top_k_per_hop=mh.top_k_per_hop,
                                           question_type="bridge")
    for r in results:
        conf_str = f" conf={r.confidence:.2f}" if r.confidence > 0 else ""
        log.info(f"[{r.rank}] hop={r.hop} score={r.score:.4f}{conf_str} title={r.passage.title}")
