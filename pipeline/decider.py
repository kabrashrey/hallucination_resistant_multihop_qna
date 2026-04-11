from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pipeline.reranker import RerankResult
from pipeline.verifier import VerificationResult


@dataclass
class DecisionOutput:
    answer: str
    supporting_passage_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0
    verifier_supported: bool = False
    support_score: float = 0.0


class Decider:
    def decide(
        self,
        answer: str,
        verification: Optional[VerificationResult],
        reranked_results: Sequence[RerankResult],
        supporting_facts: Optional[Sequence[Sequence[Any]]] = None,
        verifier: Optional["Verifier"] = None,
        retry_fn: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
        attempt_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[DecisionOutput, Dict[str, Any]]:
        attempt: Dict[str, Any] = dict(attempt_metadata or {})
        attempt.update(
            {
                "answer": answer,
                "supporting_facts": supporting_facts,
                "reranked_results": reranked_results,
                "full_response": attempt.get("full_response"),
            }
        )

        current_verification = verification
        is_supported = bool(getattr(current_verification, "is_supported", False))

        if current_verification is not None and not is_supported and verifier and retry_fn:
            retry = retry_fn()
            if retry:
                attempt.update(retry)
                current_verification = verifier.verify(
                    answer=attempt["answer"],
                    evidence=attempt["reranked_results"],
                    supporting_facts=attempt["supporting_facts"],
                )
                is_supported = bool(getattr(current_verification, "is_supported", False))

        supporting_ids = self._extract_supporting_passage_ids(
            attempt["reranked_results"], attempt["supporting_facts"]
        )
        support_score = float(getattr(current_verification, "support_score", 0.0) or 0.0)

        final_answer = (attempt["answer"] or "").strip()
        if current_verification is not None and not is_supported:
            final_answer = "Insufficient evidence"

        return (
            DecisionOutput(
                answer=final_answer,
                supporting_passage_ids=supporting_ids,
                confidence=round(support_score, 4),
                verifier_supported=is_supported,
                support_score=round(support_score, 4),
            ),
            attempt,
        )

    def _extract_supporting_passage_ids(
        self,
        reranked_results: Sequence[RerankResult],
        supporting_facts: Optional[Sequence[Sequence[Any]]],
    ) -> List[str]:
        if not reranked_results:
            return []

        if not supporting_facts:
            return []

        ids: List[str] = []
        for fact in supporting_facts:
            if not isinstance(fact, (list, tuple)) or len(fact) < 2:
                continue
            title = fact[0]
            try:
                sent_idx = int(fact[1])
            except Exception:
                continue

            for r in reranked_results:
                if r.passage.title != title:
                    continue
                if sent_idx in (r.supporting_sentence_indices or []):
                    ids.append(self._passage_identifier(r))
                    break
                if 0 <= sent_idx < len(r.passage.sentences):
                    ids.append(self._passage_identifier(r))
                    break

        # De-duplicate while preserving order
        seen = set()
        ordered = []
        for pid in ids:
            if pid and pid not in seen:
                seen.add(pid)
                ordered.append(pid)
        return ordered

    def _passage_identifier(self, result: RerankResult) -> str:
        passage = result.passage
        if getattr(passage, "passage_id", None):
            return str(passage.passage_id)
        if passage.title:
            return str(passage.title)
        return f"passage_{result.rank}"
