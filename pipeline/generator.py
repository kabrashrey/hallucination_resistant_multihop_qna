import json
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import requests

from scripts.config import load_config
from scripts.logger import get_logger

log = get_logger("generator")


@dataclass
class GeneratorOutput:
    answer: str                        
    supporting_facts: List[List[str]]  
    full_response: str                 
    model_used: str                    
    generation_time: float             

class Generator:
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        model_small: str = "mistral:7b",
        model_large: str = "mistral:7b",
        request_timeout: int = 120,
        validate_citations: bool = True,
        retry_on_parse_failure: bool = True,
    ):
        self.ollama_base_url = ollama_base_url
        self.model_small = model_small
        self.model_large = model_large
        self.request_timeout = request_timeout
        self.validate_citations = validate_citations
        self.retry_on_parse_failure = retry_on_parse_failure
        self.session = requests.Session()  # Connection pooling

        log.success(f"Generator ready. Ollama at {ollama_base_url}")

    @classmethod
    def from_config(cls, cfg=None) -> "Generator":
        if cfg is None or isinstance(cfg, (str,)):
            cfg = load_config(cfg)
        g = cfg.generator
        return cls(
            ollama_base_url=g.ollama_base_url,
            model_small=g.model_small,
            model_large=g.model_large,
            request_timeout=g.request_timeout,
            validate_citations=g.validate_citations,
            retry_on_parse_failure=getattr(g, 'retry_on_parse_failure', True),
        )

    def generate(
        self,
        prompt: str,
        target_model: str,
        temperature: float,
        supporting_fact_indices: Optional[Dict] = None,
        fact_mapping: Optional[Dict] = None,
    ) -> GeneratorOutput:
        if target_model == "complex":
            model_name = self.model_large
        else:
            model_name = self.model_small

        start_time = time.time()
        response_text = self._call_ollama(prompt, model_name, temperature)
        generation_time = time.time() - start_time

        answer, supporting_facts = self._parse_output(response_text, supporting_fact_indices, fact_mapping)

        # If parsing failed to get both answer+facts, try a repair call
        if self.retry_on_parse_failure and fact_mapping and not supporting_facts and answer:
            repaired_answer, repaired_facts = self._repair_json(
                response_text, model_name, fact_mapping, supporting_fact_indices
            )
            if repaired_facts:
                answer = repaired_answer or answer
                supporting_facts = repaired_facts
                generation_time = time.time() - start_time  # Include retry time

        return GeneratorOutput(
            answer=answer,
            supporting_facts=supporting_facts,
            full_response=response_text,
            model_used=model_name,
            generation_time=generation_time,
        )

    def generate_with_prompt(
        self,
        prompt: str,
        model_name: str,
        temperature: float = 0.3,
        supporting_fact_indices: Optional[Dict] = None,
        fact_mapping: Optional[Dict] = None,
    ) -> tuple:
        response_text = self._call_ollama(prompt, model_name, temperature)
        answer, supporting_facts = self._parse_output(response_text, supporting_fact_indices, fact_mapping)
        return answer, supporting_facts

    def _call_ollama(self, prompt: str, model: str, temperature: float) -> str:
        url = f"{self.ollama_base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
            "options": {"num_ctx": 8192},
        }

        try:
            log.step(f"Calling Ollama ({model}) at {url}...")
            response = self.session.post(
                url,
                json=payload,
                timeout=self.request_timeout,
            )
            response.raise_for_status()
        except requests.ConnectionError as e:
            log.error(f"Cannot connect to Ollama at {self.ollama_base_url}")
            raise RuntimeError(
                f"Ollama not running at {self.ollama_base_url}. "
                f"Start it with: ollama serve"
            ) from e
        except requests.Timeout as e:
            log.error(f"Ollama request timeout after {self.request_timeout}s")
            raise RuntimeError(
                f"Ollama request took too long (timeout: {self.request_timeout}s)"
            ) from e
        except requests.RequestException as e:
            log.error(f"Ollama request failed: {e}")
            raise RuntimeError(f"Ollama API error: {e}") from e

        try:
            response_data = response.json()
            response_text = response_data.get("response", "")
        except Exception as e:
            log.error(f"Failed to parse Ollama response: {e}")
            raise RuntimeError(f"Failed to parse Ollama JSON response: {e}") from e

        log.info(f"Received {len(response_text)} chars from {model}")
        return response_text

    def _parse_output(
        self, response_text: str, supporting_fact_indices: Optional[Dict] = None,
        fact_mapping: Optional[Dict] = None
    ) -> tuple:
        supporting_facts = []
        answer = ""
        log.debug(f"Raw LLM response ({len(response_text)} chars): {response_text[:500]}")

        # --- Stage 1: Try clean JSON parsing ---
        try:
            clean_text = re.sub(r'```json\s*', '', response_text)
            clean_text = re.sub(r'```\s*', '', clean_text)
            clean_text = clean_text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')

            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', clean_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)

                try:
                    parsed = json.loads(json_str)
                except json.JSONDecodeError:
                    parsed = None

                if isinstance(parsed, dict):
                    answer = parsed.get("answer", "").strip()
                    fact_numbers = parsed.get("supporting_fact_numbers", [])
                    if isinstance(fact_numbers, list) and fact_mapping:
                        hallucinated_facts = []
                        for fact_num in fact_numbers:
                            try:
                                fact_num = int(fact_num)
                                if fact_num in fact_mapping:
                                    title, sent_idx = fact_mapping[fact_num]
                                    supporting_facts.append([title, int(sent_idx)])
                                else:
                                    hallucinated_facts.append(fact_num)
                            except (ValueError, TypeError, KeyError):
                                log.debug(f"Could not map fact number {fact_num} to title/index")

                        if hallucinated_facts:
                            log.warning(f"LLM hallucinated fact numbers not in mapping: {hallucinated_facts}")

                        if answer and (supporting_facts or "cannot" in answer.lower()):
                            log.info(f"Parsed JSON (citation selection): {len(supporting_facts)} supporting facts")
                            if self.validate_citations and supporting_fact_indices:
                                self._validate_citations(supporting_facts, supporting_fact_indices)
                            return answer, supporting_facts

                    sf_list = parsed.get("supporting_facts") or parsed.get("supportiing_facts", [])

                    if isinstance(sf_list, list):
                        for sf in sf_list:
                            if isinstance(sf, list) and len(sf) >= 2:
                                try:
                                    passage_num = int(sf[0])
                                    sent_idx = int(sf[1])

                                    if supporting_fact_indices and passage_num in supporting_fact_indices:
                                        facts_in_passage = supporting_fact_indices[passage_num]
                                        if facts_in_passage:
                                            title = facts_in_passage[0][0]
                                            supporting_facts.append([title, int(sent_idx)])
                                        else:
                                            supporting_facts.append([str(passage_num), int(sent_idx)])
                                    else:
                                        supporting_facts.append([str(passage_num), int(sent_idx)])
                                except (ValueError, TypeError, IndexError):
                                    supporting_facts.append([str(sf[0]).strip(), int(sf[1])])

                    if answer and (supporting_facts or "cannot" in answer.lower()):
                        log.info(f"Parsed JSON (citation generation): {len(supporting_facts)} supporting facts")
                        if self.validate_citations and supporting_fact_indices:
                            self._validate_citations(supporting_facts, supporting_fact_indices)
                        return answer, supporting_facts
        except Exception as e:
            log.debug(f"JSON parsing failed: {e}, trying targeted extraction")

        # --- Stage 2: Targeted key extraction from broken JSON ---
        # Handles trailing commas, extra text, single quotes, etc.
        if not answer and fact_mapping:
            try:
                # Try to extract "answer" value even from malformed JSON
                answer_match = re.search(
                    r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', clean_text
                )
                if answer_match:
                    answer = answer_match.group(1).strip()
                    log.info(f"Targeted extraction recovered answer: {answer[:80]}")

                # Try to extract "supporting_fact_numbers" array
                facts_match = re.search(
                    r'"supporting_fact_numbers"\s*:\s*\[([^\]]*)\]', clean_text
                )
                if facts_match and fact_mapping:
                    nums_str = facts_match.group(1)
                    for num_str in re.findall(r'\d+', nums_str):
                        fact_num = int(num_str)
                        if fact_num in fact_mapping:
                            title, sent_idx = fact_mapping[fact_num]
                            supporting_facts.append([title, int(sent_idx)])

                if answer and (supporting_facts or "cannot" in answer.lower()):
                    log.info(f"Targeted extraction: {len(supporting_facts)} supporting facts")
                    if self.validate_citations and supporting_fact_indices:
                        self._validate_citations(supporting_facts, supporting_fact_indices)
                    return answer, supporting_facts
            except Exception as e:
                log.debug(f"Targeted key extraction failed: {e}")

        # --- Stage 3: Free-form fallback ---
        if not answer:
            log.debug("JSON + targeted extraction failed, falling back to free-form")
            answer = self._extract_answer_from_text(response_text)

        if fact_mapping and not supporting_facts:
            try:
                # Narrower patterns — only match fact-related references, not arbitrary brackets
                fact_patterns = [
                    r'\bFact\s+(\d+)',
                    r'\bfact\s+(\d+)',
                    r'fact\s+numbers?\s*:?\s*\[([^\]]+)\]',
                    r'supporting_fact.*?:?\s*\[([^\]]+)\]',
                ]

                extracted_fact_nums = set()
                for pattern in fact_patterns:
                    matches = re.findall(pattern, response_text, re.IGNORECASE)
                    for match in matches:
                        if ',' in str(match):
                            nums = [int(n.strip()) for n in str(match).split(',') if n.strip().isdigit()]
                            extracted_fact_nums.update(nums)
                        else:
                            try:
                                extracted_fact_nums.add(int(match))
                            except (ValueError, TypeError):
                                pass

                hallucinated_facts = [fn for fn in sorted(extracted_fact_nums) if fn not in fact_mapping]
                if hallucinated_facts:
                    log.warning(f"LLM hallucinated fact numbers in free-form: {hallucinated_facts}")

                for fact_num in sorted(extracted_fact_nums):
                    if fact_num in fact_mapping:
                        title, sent_idx = fact_mapping[fact_num]
                        supporting_facts.append([title, int(sent_idx)])

                if supporting_facts:
                    log.info(f"Extracted {len(supporting_facts)} supporting facts (free-form selection)")
                    if self.validate_citations and supporting_fact_indices:
                        self._validate_citations(supporting_facts, supporting_fact_indices)
            except Exception as e:
                log.debug(f"Free-form selection parsing failed: {e}")

        if "Supporting facts:" in response_text and not supporting_facts:
            try:
                sf_part = response_text.split("Supporting facts:")[1]
                if "Answer:" in sf_part:
                    sf_part = sf_part.split("Answer:")[0]

                citation_pattern = r"\[([^\]]+),\s*(\d+)\]"
                matches = re.findall(citation_pattern, sf_part)

                for title, sent_idx in matches:
                    supporting_facts.append([title.strip(), int(sent_idx)])

                if supporting_facts:
                    log.info(f"Extracted {len(supporting_facts)} supporting facts (free-form generation)")
                    if self.validate_citations and supporting_fact_indices:
                        self._validate_citations(supporting_facts, supporting_fact_indices)
            except Exception as e:
                log.debug(f"Free-form generation parsing failed: {e}")

        answer = self._normalize_answer(answer)
        return answer, supporting_facts

    def _extract_answer_from_text(self, response_text: str) -> str:
        """Extract the answer portion from raw LLM text, stripping reasoning and evidence echoes."""
        text = response_text.strip()

        # If the LLM echoed back the evidence block, strip it
        if "Evidence:" in text and "QUESTION:" in text:
            # Take only the part after the last QUESTION or TASK marker
            for marker in ["RESPOND WITH", "TASK:", "RULES:"]:
                if marker in text:
                    text = text.split(marker)[-1]

        # Look for explicit answer patterns
        answer_patterns = [
            r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"',
            r'(?:The\s+)?answer\s+is:?\s*(.+?)(?:\.|$)',
            r'Answer:\s*(.+?)(?:\n|$)',
        ]
        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Last resort: take first non-empty line that isn't a fact reference
        for line in text.split('\n'):
            line = line.strip()
            if line and not line.startswith('Fact') and not line.startswith('[') and len(line) > 2:
                return line

        return text

    def _repair_json(
        self, raw_response: str, model_name: str,
        fact_mapping: Dict, supporting_fact_indices: Optional[Dict] = None
    ) -> tuple:
        """Make a lightweight second LLM call to reformat a failed JSON response."""
        try:
            # Truncate to avoid context overflow — just need the answer portion
            truncated = raw_response[:500]
            repair_prompt = (
                f"Extract the answer and supporting fact numbers from this text.\n"
                f"Respond ONLY with valid JSON, nothing else:\n"
                f'{{\"supporting_fact_numbers\": [0, 1, 2], \"answer\": \"your answer\"}}\n\n'
                f"Text: {truncated}"
            )

            log.info("JSON repair: making second LLM call to reformat response...")
            repair_response = self._call_ollama(repair_prompt, model_name, temperature=0.0)

            # Parse the repair response (reuse existing logic but without recursive retry)
            answer, supporting_facts = self._parse_output(
                repair_response, supporting_fact_indices, fact_mapping
            )

            if supporting_facts:
                log.info(f"JSON repair recovered {len(supporting_facts)} supporting facts")
            return answer, supporting_facts

        except Exception as e:
            log.debug(f"JSON repair failed: {e}")
            return "", []

    def _normalize_answer(self, answer: str) -> str:
        """Normalize LLM answers to improve EM matching against HotpotQA gold."""
        if not answer:
            return answer

        # Strip whitespace and surrounding quotes
        answer = answer.strip().strip('"').strip("'").strip()

        # Remove common LLM preamble patterns
        preamble_patterns = [
            r'^Based on the (?:provided )?evidence,?\s*',
            r'^According to the (?:provided )?(?:evidence|facts|passages),?\s*',
            r'^The answer is:?\s*',
            r'^Answer:?\s*',
            r'^From the evidence,?\s*',
        ]
        for pattern in preamble_patterns:
            answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)

        # Remove trailing periods from short answers (common LLM habit)
        if len(answer.split()) <= 5 and answer.endswith('.'):
            answer = answer[:-1].strip()

        # Normalize yes/no/noanswer to lowercase (HotpotQA eval normalizes these)
        answer_lower = answer.lower().strip()
        if answer_lower in ('yes', 'no', 'noanswer', 'yes.', 'no.'):
            answer = answer_lower.rstrip('.')

        return answer.strip()

    def _validate_citations(
        self, supporting_facts: List[List[str]], supporting_fact_indices: Dict
    ) -> None:
        valid_facts = set()
        for _, facts in supporting_fact_indices.items():
            for title, sent_idx in facts:
                valid_facts.add((title, int(sent_idx)))

        for fact in supporting_facts:
            title, sent_idx = fact[0], int(fact[1])
            if (title, sent_idx) not in valid_facts:
                log.warning(
                    f"Citation [{title}, {sent_idx}] not in supporting facts indices. "
                    f"LLM may have hallucinated this fact."
                )

    def __repr__(self) -> str:
        return (
            f"Generator(ollama={self.ollama_base_url}, "
            f"small={self.model_small}, large={self.model_large}, "
            f"validate_citations={self.validate_citations})"
        )

if __name__ == "__main__":
    from pipeline.data_loader import HotpotQALoader
    from pipeline.indexer import HybridRetriever
    from pipeline.prompt_builder import PromptBuilder
    from pipeline.reranker import Reranker
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

    log.info(f"Building retriever with {len(passages)} passages...")
    retriever = HybridRetriever.from_config(cfg)
    retriever.index(passages, show_progress=False)

    log.info("Loading reranker...")
    reranker = Reranker.from_config(cfg)

    log.info("Initializing prompt builder...")
    pb = PromptBuilder.from_config(cfg)

    log.info("Initializing generator...")
    gen = Generator.from_config(cfg)

    for i, ex in enumerate(examples[:3]):
        log.info(f"\n{'='*80}")
        log.info(f"Example {i}: {ex.id}")
        log.info(f"Question: {ex.question}")
        log.info(f"Gold answer: {ex.answer}")

        retrieved = retriever.retrieve_multihop(
            ex.question, hops=2, top_k=20, question_type=ex.question_type
        )
        log.info(f"Retrieved {len(retrieved)} candidates")

        reranked = reranker.rerank(ex.question, retrieved, top_k=5)
        log.info(f"Re-ranked to {len(reranked)} passages")

        pb_output = pb.build(ex.question, reranked)
        log.info(f"Complexity score: {pb_output.complexity_score:.3f}")
        log.info(f"Target model: {pb_output.target_model}")

        gen_output = gen.generate(
            prompt=pb_output.prompt,
            target_model=pb_output.target_model,
            temperature=(
                cfg.prompt_builder.temperature_large_model
                if pb_output.target_model == "complex"
                else cfg.prompt_builder.temperature_small_model
            ),
            supporting_fact_indices=pb_output.supporting_fact_indices,
        )

        log.info(f"\n{'='*80}")
        log.info(f"Generated answer: {gen_output.answer}")
        log.info(f"Supporting facts: {gen_output.supporting_facts}")
        log.info(f"Model: {gen_output.model_used}")
        log.info(f"Generation time: {gen_output.generation_time:.2f}s")
        log.info(f"\n{'='*80}")
