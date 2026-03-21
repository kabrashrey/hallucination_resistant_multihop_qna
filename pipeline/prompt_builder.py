from dataclasses import dataclass
from typing import List, Dict

from pipeline.data_loader import Passage, SupportingFact
from pipeline.indexer import RetrievalResult
from pipeline.reranker import RerankResult
from scripts.config import load_config
from scripts.logger import get_logger

log = get_logger("prompt_builder")

BRIDGE_KEYWORDS = {"who", "which", "when", "where", "what", "how", "portrayed", "actor", "character", "played"}


@dataclass
class PromptBuilderOutput:
    prompt: str                           
    complexity_score: float               
    target_model: str                     
    passage_titles: List[str]             
    supporting_fact_indices: Dict         
    fact_mapping: Dict                    
    metadata: Dict                        


class PromptBuilder:
    def __init__(self, cfg=None):
        if cfg is None or isinstance(cfg, str):
            cfg = load_config(cfg).prompt_builder
        self.cfg = cfg
        log.success("PromptBuilder initialized.")

    @classmethod
    def from_config(cls, cfg=None) -> "PromptBuilder":
        if cfg is None or isinstance(cfg, (str,)):
            cfg = load_config(cfg)
        return cls(cfg.prompt_builder)

    def build(
        self,
        query: str,
        reranked_results: List[RerankResult],
        include_metadata: bool = True,
        use_citation_selection: bool = True,
    ) -> PromptBuilderOutput:
        if not reranked_results:
            raise ValueError("No reranked results provided")

        # Sort by reranker score (highest first) for better LLM prioritization
        reranked_results = sorted(reranked_results, key=lambda r: r.score, reverse=True)

        passage_titles = [r.passage.title for r in reranked_results]
        num_passages = len(reranked_results)
        evidence_block = self._format_evidence_block(reranked_results)
        complexity_score = self._compute_complexity_score(query, reranked_results)

        supporting_fact_indices = self._extract_supporting_fact_indices(reranked_results)
        facts_list_str, fact_mapping = self._format_facts_list(reranked_results)

        # Ensure complexity_routing_threshold is a float (defensive cast for config loading issues)
        threshold = float(self.cfg.complexity_routing_threshold) if hasattr(self.cfg, 'complexity_routing_threshold') else 0.6
        target_model = "complex" if complexity_score > threshold else "simple"

        if use_citation_selection:
            instructions = self._build_instructions(query, facts_list_str=facts_list_str, fact_mapping=fact_mapping)
        else:
            instructions = self._build_instructions(query)

        if self.cfg.evidence_first:
            if use_citation_selection:
                prompt = f"{evidence_block}\n\n{instructions}"
            else:
                prompt = f"{evidence_block}\n\n{instructions}"
        else:
            if use_citation_selection:
                prompt = f"{instructions}\n\n{evidence_block}"
            else:
                prompt = f"{instructions}\n\n{evidence_block}"

        if len(prompt) > self.cfg.max_evidence_chars:
            log.warning(f"Prompt too long ({len(prompt)} chars), truncating to {self.cfg.max_evidence_chars}")
            prompt = prompt[:self.cfg.max_evidence_chars] + "\n[... truncated ...]"

        metadata = {} if not include_metadata else {
            "num_passages": num_passages,
            "prompt_length": len(prompt),
            "complexity_raw": complexity_score,
            "total_sentences": sum(len(r.supporting_sentences) for r in reranked_results),
            "avg_retrieval_score": sum(r.retrieval_score for r in reranked_results) / num_passages,
            "avg_rerank_score": sum(r.score for r in reranked_results) / num_passages,
            "num_facts": len(fact_mapping),
        }

        return PromptBuilderOutput(
            prompt=prompt,
            complexity_score=complexity_score,
            target_model=target_model,
            passage_titles=passage_titles,
            supporting_fact_indices=supporting_fact_indices,
            fact_mapping=fact_mapping,
            metadata=metadata,
        )

    def _format_evidence_block(self, results: List[RerankResult]) -> str:
        lines = ["Evidence:"]

        for passage_idx, result in enumerate(results):
            passage = result.passage
            # Include relevance score (0-1) in passage header to guide LLM
            relevance = max(0.0, min(1.0, result.score))  # Clamp to [0, 1]
            lines.append(f"Passage {passage_idx}: [{passage.title} (relevance: {relevance:.2f})]")

            for sent_str, sent_idx in zip(result.supporting_sentences, result.supporting_sentence_indices):
                display_sent = sent_str[:200] + "..." if len(sent_str) > 200 else sent_str
                lines.append(f"  [{sent_idx}] {display_sent}")

        return "\n".join(lines)

    def _compute_complexity_score(self, query: str, results: List[RerankResult]) -> float:
        score = 0.0

        # Factor 1: Query length
        has_long_query = len(query) > self.cfg.complexity_length_threshold
        score += self.cfg.complexity_length_weight * float(has_long_query)

        # Factor 2: Bridge keywords
        query_lower = query.lower()
        has_bridge_kw = any(kw in query_lower for kw in BRIDGE_KEYWORDS)
        score += self.cfg.complexity_keywords_weight * float(has_bridge_kw)

        # Factor 3: Confidence (how ambiguous is the retrieval?)
        # Use reranker top-score gap as a better proxy than raw retrieval scores
        if len(results) >= 2:
            top = float(results[0].score)
            second = float(results[1].score)
            denom = abs(top) + 1e-6
            gap = max(min((top - second) / denom, 1.0), 0.0)
            confidence_penalty = 1.0 - gap
        else:
            confidence_penalty = 0.5
        score += self.cfg.complexity_confidence_weight * confidence_penalty

        # Factor 4: Supporting sentence count (many sentences = complex reasoning)
        total_sentences = sum(len(r.supporting_sentences) for r in results)
        has_many_sentences = total_sentences > self.cfg.complexity_sentence_threshold
        score += self.cfg.complexity_sentences_weight * float(has_many_sentences)

        return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]

    def _extract_supporting_fact_indices(self, results: List[RerankResult]) -> Dict:
        fact_indices = {}

        for passage_idx, result in enumerate(results):
            facts = [
                (result.passage.title, sent_idx)
                for sent_idx in result.supporting_sentence_indices
            ]
            fact_indices[passage_idx] = facts

        return fact_indices

    def _format_facts_list(self, results: List[RerankResult]) -> str:
        lines = ["AVAILABLE FACTS:"]
        fact_mapping = {}  # fact_number -> (title, sentence_idx)
        fact_number = 0

        for result in results:
            passage = result.passage
            for sent_str, sent_idx in zip(result.supporting_sentences, result.supporting_sentence_indices):
                display_sent = sent_str[:200] + "..." if len(sent_str) > 200 else sent_str
                lines.append(f"Fact {fact_number}: [{passage.title}, {sent_idx}] {display_sent}")
                fact_mapping[fact_number] = (passage.title, sent_idx)
                fact_number += 1

        return "\n".join(lines), fact_mapping

    def _build_instructions(self, query: str, facts_list_str: str = "", fact_mapping: Dict = None) -> str:
        if fact_mapping is not None and facts_list_str:
            # Citation selection approach
            instructions = f"""QUESTION: {query}

{facts_list_str}

EXAMPLES:

Example 1 (bridge question):
AVAILABLE FACTS:
Fact 0: [The Departed, 0] The Departed is a 2006 American crime film directed by Martin Scorsese.
Fact 1: [The Departed, 2] The film stars Leonardo DiCaprio, Matt Damon, and Jack Nicholson.
Fact 2: [Martin Scorsese, 0] Martin Scorsese is an American film director born in 1942.
Fact 3: [Martin Scorsese, 1] He was born in Queens, New York City.

Question: Where was the director of The Departed born?
{{"reasoning": "Fact 0 identifies Martin Scorsese as the director. Fact 3 states he was born in Queens, New York City.", "supporting_fact_numbers": [0, 2, 3], "answer": "Queens, New York City"}}

Example 2 (comparison question, entity answer):
AVAILABLE FACTS:
Fact 0: [Oceanview High, 0] Oceanview High School was founded in 1965.
Fact 1: [Oceanview High, 1] It is located in Santa Monica, California.
Fact 2: [Ridgemont Academy, 0] Ridgemont Academy was established in 1948.
Fact 3: [Ridgemont Academy, 2] The school has won multiple state championships.

Question: Which school was founded first, Oceanview High or Ridgemont Academy?
{{"reasoning": "Fact 0 says Oceanview High was founded in 1965. Fact 2 says Ridgemont Academy was established in 1948. 1948 is earlier than 1965.", "supporting_fact_numbers": [0, 2], "answer": "Ridgemont Academy"}}

Example 3 (comparison question, yes/no answer):
AVAILABLE FACTS:
Fact 0: [Silver Hawks, 0] The Silver Hawks are a rock band formed in London in 2003.
Fact 1: [Silver Hawks, 1] The band consists of four members.
Fact 2: [The Embers, 0] The Embers are an indie group from Manchester.
Fact 3: [The Embers, 1] They have four members and formed in 2001.

Question: Do the Silver Hawks and The Embers have the same number of members?
{{"reasoning": "Fact 1 says Silver Hawks has four members. Fact 3 says The Embers have four members. They are equal.", "supporting_fact_numbers": [1, 3], "answer": "yes"}}

TASK: Answer the question using ONLY the available facts.

RESPOND WITH THIS JSON FORMAT - NOTHING ELSE:
{{"reasoning": "Brief chain of thought", "supporting_fact_numbers": [0, 1, 3], "answer": "entity or yes/no"}}

RULES:
- Output ONLY the JSON block, no other text before or after.
- "reasoning": briefly explain how the facts connect to your answer.
- "supporting_fact_numbers": list ALL fact numbers that support your answer, including bridging facts.
- "answer": 1-5 words MAXIMUM. Extract the specific entity, name, date, or yes/no. Never use full sentences.
  BAD: "The film was directed by Martin Scorsese" GOOD: "Martin Scorsese"
  BAD: "decisive defeat for Italy and secured Ethiopian sovereignty" GOOD: "Battle of Adwa"
- For yes/no questions, answer "yes" or "no" (lowercase only).
- For "who/which is older/larger/first" questions, answer with the ENTITY NAME, not "yes" or "no".
- Give exactly ONE answer — the single most specific entity. NEVER list multiple answers separated by commas.
  BAD: "United States Ambassador, Chief of Protocol" GOOD: "Chief of Protocol"
- Do NOT add parenthetical clarifications or alternative names.
  BAD: "Kansas Song (We're From Kansas)" GOOD: "Kansas Song"
  BAD: "3,677 seated (4,000 capacity)" GOOD: "3,677 seated"
- You MUST provide an answer using the available facts. Even if you only have partial evidence, give your best logical guess.
- NEVER output "Cannot determine from evidence", always make a prediction.
"""
        else:
            instructions = f"""Question: {query}

Instructions:
1. Read the evidence above carefully.
2. Identify which evidence passages (Passage 0, 1, 2, etc.) are relevant to the question.
3. Reason step-by-step using ONLY the evidence provided.
4. List the supporting facts by citing the passage number and sentence index: [passage_num, sentence_idx].
5. Generate a clear, concise answer based on the evidence.

Output Format (JSON):
{{
  "supporting_facts": [[0, 0], [1, 3], [2, 1]],
  "answer": "Your answer here"
}}

Important:
- Output ONLY valid JSON, no additional text before or after.
- supporting_facts is a list of [passage_number, sentence_index] pairs (NOT title).
- Passage numbers are 0-4 as shown in the evidence block above.
- If the answer cannot be determined from the evidence, set answer to "Cannot answer based on the provided evidence." with empty supporting_facts.
"""
        return instructions

    def __repr__(self) -> str:
        return (
            f"PromptBuilder(evidence_first={self.cfg.evidence_first}, "
            f"complexity_threshold={self.cfg.complexity_routing_threshold}, "
            f"max_chars={self.cfg.max_evidence_chars})"
        )


if __name__ == "__main__":
    from pipeline.data_loader import HotpotQALoader
    from pipeline.indexer import HybridRetriever
    from pipeline.reranker import Reranker

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

    log.info(f"Building retriever with {len(passages)} passages...")
    retriever = HybridRetriever.from_config(cfg)
    retriever.index(passages, show_progress=False)

    log.info("Loading reranker...")
    reranker = Reranker.from_config(cfg)

    log.info("Initializing prompt builder...")
    pb = PromptBuilder.from_config(cfg)

    ex = examples[0]
    log.info(f"\n{'='*80}")
    log.info(f"Example: {ex.id}")
    log.info(f"Question: {ex.question}")
    log.info(f"Gold answer: {ex.answer}")
    log.info(f"Gold SP: {[(sf.title, sf.sentence_index) for sf in ex.supporting_facts]}")

    retrieved = retriever.retrieve_multihop(
        ex.question, hops=2, top_k=20, question_type=ex.question_type
    )
    log.info(f"\nRetrieved {len(retrieved)} candidates")

    reranked = reranker.rerank(ex.question, retrieved, top_k=5)
    log.info(f"Re-ranked to {len(reranked)} passages")

    output = pb.build(ex.question, reranked)

    log.info(f"\n{'='*80}")
    log.info(f"Complexity score: {output.complexity_score:.3f}")
    log.info(f"Target model: {output.target_model}")
    log.info(f"Prompt length: {len(output.prompt)} chars")

    log.info(f"\n{'='*80}")
    log.info("PROMPT:")
    log.info(f"{'='*80}")
    print(output.prompt)

    log.info(f"\n{'='*80}")
    log.info("Supporting fact indices (for parsing LLM output):")
    for passage_idx, facts in output.supporting_fact_indices.items():
        log.info(f"  Passage {passage_idx}: {facts}")
