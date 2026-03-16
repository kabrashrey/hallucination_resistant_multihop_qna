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
    ):
        self.ollama_base_url = ollama_base_url
        self.model_small = model_small
        self.model_large = model_large
        self.request_timeout = request_timeout
        self.validate_citations = validate_citations
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
                        # Citation selection: map fact numbers back to [title, sent_idx]
                        for fact_num in fact_numbers:
                            try:
                                fact_num = int(fact_num)
                                if fact_num in fact_mapping:
                                    title, sent_idx = fact_mapping[fact_num]
                                    supporting_facts.append([title, int(sent_idx)])
                            except (ValueError, TypeError, KeyError):
                                log.debug(f"Could not map fact number {fact_num} to title/index")

                        if answer and (supporting_facts or "cannot" in answer.lower()):
                            log.info(f"Parsed JSON (citation selection): {len(supporting_facts)} supporting facts")
                            if self.validate_citations and supporting_fact_indices:
                                self._validate_citations(supporting_facts, supporting_fact_indices)
                            return answer, supporting_facts

                    sf_list = parsed.get("supporting_facts") or parsed.get("supportiing_facts", []) # common LLM typo

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
            log.debug(f"JSON parsing failed: {e}, trying fallback format")
        
        answer = response_text.strip()

        if fact_mapping:
            try:
                fact_patterns = [
                    r'\bFact\s+(\d+)',                    
                    r'\bfact\s+(\d+)',                    
                    r'fact\s+numbers?\s*:?\s*\[([^\]]+)\]',
                    r'supporting_fact.*?:?\s*\[([^\]]+)\]',
                    r'\[(\d+(?:,\s*\d+)*)\]',             
                ]

                extracted_fact_nums = set()
                for pattern in fact_patterns:
                    matches = re.findall(pattern, response_text, re.IGNORECASE)
                    for match in matches:
                        if ',' in str(match):
                            nums = [int(n.strip()) for n in str(match).split(',')]
                            extracted_fact_nums.update(nums)
                        else:
                            try:
                                extracted_fact_nums.add(int(match))
                            except (ValueError, TypeError):
                                pass

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

        return answer, supporting_facts

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
