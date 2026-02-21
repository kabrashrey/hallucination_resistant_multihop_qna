"""
Hybrid Multi-Hop Retriever: BM25 + FAISS
The hybrid approach covers both cases — BM25 catches exact term matches that dense might miss, and dense catches semantic similarity that BM25 can't understand.

1. Sparse retriever: BM25
2. Dense retriever: FAISS
3. Reciprocal Rank Fusion (RRF)
4. Candidate Pooling
5. Confidence Estimation
6. Multi-hop itertive retrieval
7. Bridge entity extraction
8. Save/load support

Usage:
    from pipeline.data_loader import HotpotQALoader
    from pipeline.indexer import HybridRetriever

    loader = HotpotQALoader("data/hotpot_dev_distractor_v1.json")
    passages = loader.get_all_passages()

    retriever = HybridRetriever(embed_model="all-MiniLM-L6-v2")
    retriever.index(passages)
    retriever.save("index_cache")       # save to disk
    retriever.load("index_cache")       # load from disk (skip re-encoding)
    results = retriever.retrieve("Who directed Doctor Strange?", top_k=5)

    # Multi-hop retrieval for bridge questions
    results = retriever.retrieve_multihop("What position was held by the woman who played Corliss Archer?", hops=2, top_k=5)
"""

import os
# Prevent FAISS segfaults on macOS must be set before importing FAISS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import faiss
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from pipeline.data_loader import Passage


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
    """
    Dense retriever using SentenceTransformers + FAISS --> Neural semantic retrieval
    Encodes passages into vectors using neural network, uses FAISS for fast nearest-neighbor search
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu", batch_size: int = 64):
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name, device="cpu") # forced CPU, since MPS causes error with FAISS
        self.batch_size = batch_size
        self._index: Optional[faiss.IndexFlatIP] = None  # inner-product on unit vector = cosine
        self._dim: int = 0 # embedding dimension (384 for MiniLM)

    def index(self, texts: List[str], show_progress: bool = True):
        """
        Encode all passages and build FAISS index
        texts: list of passage strings (title + text)
        """
        # Encode in batches --> produces a (batch_size, 384) matrix, Normalize = True --> makes each vector unit length
        all_embeddings = []
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding passages")

        for i in iterator:
            batch = texts[i:i + self.batch_size]
            embeddings = self.encoder.encode(batch, convert_to_numpy=True,
                                             show_progress_bar=False, normalize_embeddings=True)
            all_embeddings.append(embeddings)

        # concatenate all batches into one big (n_passages, 384) matrix
        vectors = np.vstack(all_embeddings).astype(np.float32)
        self._dim = vectors.shape[1]

        # Build FAISS index (inner product = cosine similarity since vectors are normalized)
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(vectors)

    def score(self, query: str) -> np.ndarray:
        """
        Score all passages for a query using cosine similarity via FAISS
        Returns: array of shape (n_passages,)
        """
        if self._index is None:
            raise RuntimeError("Call .index() first")

        q_vec = self.encoder.encode([query], convert_to_numpy=True,
                                    normalize_embeddings=True).astype(np.float32)
        # Search all passages (k = total), returns top-n results sorted by cosine similarity
        n = self._index.ntotal
        distances, indices = self._index.search(q_vec, n)   # Can use top_k instead of n to speed up for FullWiki

        # Map back to original order, since FAISS returns results sorted by similarity
        scores = np.zeros(n, dtype=np.float32)
        scores[indices[0]] = distances[0]
        return scores

    def search_top_k(self, query: str, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        FAST top-k search without scoring all passages --> FAISS can stop early instead of scoring all passages
        Returns: (scores, indices) arrays of shape (top_k,)
        """
        if self._index is None:
            raise RuntimeError("Call .index() first")

        q_vec = self.encoder.encode([query], convert_to_numpy=True,
                                    normalize_embeddings=True).astype(np.float32)
        distances, indices = self._index.search(q_vec, top_k)
        return distances[0], indices[0]

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query into a vector (useful for multi-hop)"""
        return self.encoder.encode([query], convert_to_numpy=True,
                                   normalize_embeddings=True).astype(np.float32)[0]

    def save(self, path: Path):
        """Save FAISS index and metadata to disk"""
        faiss.write_index(self._index, str(path / "faiss.index"))
        with open(path / "dense_meta.json", "w") as f:
            json.dump({"dim": self._dim, "model_name": self.model_name}, f)

    def load(self, path: Path):
        """Load FAISS index from disk (no re-encoding needed)"""
        self._index = faiss.read_index(str(path / "faiss.index"))
        with open(path / "dense_meta.json", "r") as f:
            meta = json.load(f)
        self._dim = meta["dim"]

    @property
    def is_indexed(self) -> bool:
        return self._index is not None


class HybridRetriever:
    """
    Combining BM25 (sparse) and FAISS (dense) with Reciprocal Rank Fusion
    Final score = alpha * rrf_dense + (1 - alpha) * rrf_bm25

      - Per-type alpha (different weights for bridge vs comparison questions)
      - Multi-hop iterative retrieval for bridge questions
      - Save/load indices to disk

    Eg:
        retriever = HybridRetriever(embed_model="all-MiniLM-L6-v2")
        retriever.index(passages)
        retriever.save("index_cache")

        # Single-hop
        results = retriever.retrieve("Who directed Doctor Strange?", top_k=5)

        # Multi-hop (for bridge questions)
        results = retriever.retrieve_multihop("What position was held by the actress in Kiss and Tell?", hops=2)
    """

    def __init__(self, embed_model: str = "all-MiniLM-L6-v2",
                 alpha: float = 0.7,
                 alpha_bridge: Optional[float] = None,
                 alpha_comparison: Optional[float] = None,
                 rrf_k: int = 20,
                 device: str = "cpu", batch_size: int = 64):
        """
        embed_model: SentenceTransformer model name for dense retrieval (default all-MiniLM-L6-v2)
        alpha: default weight for dense scores (1-alpha = weight for BM25)
        alpha_bridge: override alpha for bridge questions (needs multi-hop, favor dense)
        alpha_comparison: override alpha for comparison questions (entities are explicit, favor BM25)
        rrf_k: RRF constant (default 60, higher = smoother ranking)
        device: torch device for encoding ("cpu", "cuda", "mps")
        batch_size: batch size for encoding passages (default 64)

        # ALPHA MEANING
        alpha = 1.0 → pure dense retrieval, only semantic similarity
        alpha = 0.7 → 70% dense, 30% BM25
        alpha = 0.5 → equal balance
        alpha = 0.3 → 30% dense, 70% BM25
        alpha = 0.0 → pure BM25 retrieval, only keyword matching
        """
        self.alpha = alpha  # dense weight, controls balance between the two retrievers
        self.alpha_bridge = alpha_bridge           # None = use default alpha
        self.alpha_comparison = alpha_comparison   # None = use default alpha
        self.rrf_k = rrf_k

        self.bm25 = BM25Retriever() # sparse component
        self.dense = DenseRetriever(model_name=embed_model, device=device,
                                    batch_size=batch_size) # dense component
        self._passages: List[Passage] = []  # list of Passage objects
        self._texts: List[str] = [] # list of passage strings

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

        print(f"Indexing {len(self._texts)} passages...")

        # Build both indices
        print("Building BM25 index...")
        self.bm25.index(self._texts)

        print("Building FAISS dense index...")
        self.dense.index(self._texts, show_progress=show_progress)

        print(f"{len(self._texts)} passages indexed.")

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

        print(f"Saved index to {save_dir}/ ({len(self._passages)} passages)")

    def load(self, save_dir: Union[str, Path]):
        """
        Load indices + passages from disk (skips encoding)
        The SentenceTransformer model is still loaded for query encoding
        """
        save_dir = Path(save_dir)
        if not (save_dir / "faiss.index").exists():
            raise FileNotFoundError(f"No saved index in {save_dir}")

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

        print(f"Loaded index from {save_dir}/ ({len(self._passages)} passages)")

    def retrieve(self, query: str, top_k: int = 10,
                 question_type: Optional[str] = None,
                 candidate_pool_size: int = 100) -> List[RetrievalResult]:
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
        pool_size = min(candidate_pool_size, len(self._passages))

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
          - Multi-word names:   "Shirley Temple", "New York City"
          - Acronyms:           "NASA", "FBI", "UNESCO"
          - Single-name entities: "Cher", "Madonna", "Beyoncé"
          - Titles with punctuation: "Dr. Strange", "St. Louis"
        """
        import re
        question_lower = question.lower()
        text = passage.text

        entities = []

        # Multi-word capitalized phrases ("Shirley Temple", "New York City") and titles with punctuation: "Dr. Strange", "St. Louis"
        entities += re.findall(r'\b([A-Z][a-z]*\.?\s+(?:[A-Z][a-z]*\.?\s*)+)', text)

        # Acronyms — 2+ consecutive uppercase letters ("NASA", "FBI", "MCU")
        entities += re.findall(r'\b([A-Z]{2,})\b', text)

        # Single capitalized words that appear mid-sentence, exclude sentence starters by requiring a lowercase letter or punctuation before
        entities += re.findall(r'(?<=[a-z.,;:!?]\s)([A-Z][a-zà-ÿ]{2,})\b', text)

        # Also include the passage title itself as a candidate
        candidates = [passage.title] + [e.strip() for e in entities]

        # Filter out entities already in the question
        bridge = []
        seen = set()
        for entity in candidates:
            entity_clean = entity.strip().rstrip('.')
            if (entity_clean.lower() not in question_lower
                    and entity_clean not in seen
                    and len(entity_clean) > 1):
                seen.add(entity_clean)
                bridge.append(entity_clean)

        return bridge[:5]  # limit to top 5 entities

    def retrieve_multihop(self, query: str, hops: int = 2, top_k: int = 10,
                          top_k_per_hop: int = 5,
                          question_type: Optional[str] = None) -> List[RetrievalResult]:
        """
        Multi-hop iterative retrieval for bridge questions

        How it works:
          Hop 1: retrieve using the original question
          Hop 2: extract named entities from hop 1 passages (not just titles!)
                 reformulate query = question + bridge entities, retrieve again
          Finally: merge, deduplicate, re-rank results from all hops

        query: the original question
        hops: number of retrieval iterations (default 2 for bridge questions)
        top_k: total results to return after merging
        top_k_per_hop: how many passages to retrieve at each hop
        question_type: "bridge" or "comparison"

        Returns: deduplicated, re-ranked RetrievalResult list

        Eg for "What position was held by the woman who portrayed Corliss Archer?":
          Hop 1 query: "What position was held by the woman who portrayed Corliss Archer?"
          Hop 1 finds: "Kiss and Tell" passage --> mentions "Shirley Temple"
          Hop 2 query: "What position was held by the woman who portrayed Corliss Archer? Shirley Temple"
          Hop 2 finds: "Shirley Temple" passage --> mentions "Chief of Protocol"
        """
        all_results: List[RetrievalResult] = []
        seen_keys: set = set()
        current_query = query

        for hop in range(hops):
            hop_results = self.retrieve(current_query, top_k=top_k_per_hop,
                                        question_type=question_type)

            # Compute confidence for this hop (how dominant is the top result?)
            hop_scores = [r.score for r in hop_results]
            hop_confidence = _compute_confidence(hop_scores)

            # Collect new (unseen) passages and extract bridge entities
            bridge_entities = []
            new_in_hop = 0
            for r in hop_results:
                key = (r.passage.title, r.passage.text)
                if key not in seen_keys:
                    seen_keys.add(key)
                    r.hop = hop
                    r.confidence = hop_confidence if new_in_hop == 0 else 0.0  # top result per hop gets confidence
                    new_in_hop += 1
                    all_results.append(r)

                    # Extract entities from passage content (not just title)
                    entities = self._extract_bridge_entities(r.passage, query)
                    bridge_entities.extend(entities)

            # Reformulate query for next hop using extracted entities
            if hop < hops - 1 and bridge_entities:
                # Deduplicate and take top entities
                unique_entities = list(dict.fromkeys(bridge_entities))[:3]
                entity_str = " ".join(unique_entities)
                current_query = f"{query} {entity_str}"

        # Re-rank all collected results by score (descending)
        all_results.sort(key=lambda r: r.score, reverse=True)

        # Re-assign ranks (keep per-hop confidence as-is)
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

    # Load a small subset
    loader = HotpotQALoader("data/hotpot_dev_distractor_v1.json") # CHANGE THIS TO FULLWIKI DATASET
    examples = loader.load(limit=50)

    # Collect unique passages from those examples
    seen = set()
    passages: List[Passage] = []
    for ex in examples:
        for ctx in ex.contexts:
            key = (ctx.title, ctx.text)
            if key not in seen:
                seen.add(key)
                passages.append(ctx)

    print(f"\n{len(passages)} unique passages from {len(examples)} examples\n")

    # Build hybrid index with per-type alpha
    retriever = HybridRetriever(
        embed_model="all-MiniLM-L6-v2",
        alpha=0.7,
        alpha_bridge=0.85,       # bridge: favor dense (needs semantic understanding)
        alpha_comparison=0.5,    # comparison: equal weight (entities are explicit)
    )
    retriever.index(passages)

    # Save to disk
    retriever.save("index_cache")
    print("Saved.\n")

    # Load from disk (no re-encoding)
    retriever2 = HybridRetriever(embed_model="all-MiniLM-L6-v2")
    retriever2.load("index_cache")
    print()

    # Test single-hop (comparison) — with confidence
    q1 = "Were Scott Derrickson and Ed Wood of the same nationality?"
    print(f"Q (comparison): {q1}")
    results = retriever2.retrieve(q1, top_k=3, question_type="comparison")
    for r in results:
        conf_str = f" conf={r.confidence:.2f}" if r.confidence > 0 else ""
        print(f"[{r.rank}] score={r.score:.4f}{conf_str} title={r.passage.title}")

    # Test multi-hop (bridge) — shows entity extraction + hop info
    q2 = "What government position was held by the woman who portrayed Corliss Archer?"
    print(f"\nQ (bridge, multi-hop): {q2}")
    results = retriever2.retrieve_multihop(q2, hops=2, top_k=5, question_type="bridge")
    for r in results:
        conf_str = f" conf={r.confidence:.2f}" if r.confidence > 0 else ""
        print(f"[{r.rank}] hop={r.hop} score={r.score:.4f}{conf_str} title={r.passage.title}")
