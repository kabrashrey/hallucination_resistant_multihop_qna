"""
HotpotQA Dataset Loader
Usage:
    from pipeline.data_loader import HotpotQALoader

    loader = HotpotQALoader("data/hotpot_dev_distractor_v1.json")
    examples = loader.load()           # list of standardized dicts
    index = loader.build_passage_index()  # {passage_id: {title, sentences, text}}
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Union

"""
@dataclass -- automatically generates __init__, __repr__, __eq__ etc. for you

####INSTEAD OF THIS:
class Passage:
    def __init__(self, title, sentences, passage_id=None):
        self.title = title
        self.sentences = sentences
        self.passage_id = passage_id

    def __repr__(self):
        return f"Passage(title={self.title})"

####YOU CAN SIMPLY WRITE THIS:
from dataclasses import dataclass

@dataclass
class Passage:
    title: str
    sentences: list
    passage_id: str = None
"""

@dataclass
class Passage:
    """
    Eg:
    title: "Scott Derrickson"
    sentences: ["Scott Derrickson (born July 16, 1966)...", "He lives in..."]
    passage_id: "Scott Derrickson::1" (set when indexing)
    """
    title: str 
    sentences: List[str]
    passage_id: Optional[str] = None

    @property # instead of p.text() we can do p.text
    def text(self) -> str:
        """Full passage text (sentences joined)"""
        return " ".join(s.strip() for s in self.sentences if isinstance(s, str))

    @property
    def title_text(self) -> str:
        """Title prepended to passage text (useful for retrieval)"""
        title  = (self.title or "").strip()
        if title:
            return f"{title} {self.text}"
        return self.text


@dataclass
class SupportingFact:
    """
    Eg:
        title: "Scott Derrickson"
        sentence_index: 0 (first sentence of that passage)
    """
    title: str
    sentence_index: int


@dataclass
class HotpotQAExample:
    """
    Eg:
        id: "1"
        question: "What is the name of the actor who played Thor in the Marvel Cinematic Universe?"
        answer: "Chris Hemsworth"
        contexts: [Passage("Avengers", ["Chris Hemsworth played Thor in the Marvel Cinematic Universe."])]
        supporting_facts: [SupportingFact("Avengers", 0)]
        question_type: "bridge"
        level: "easy"
    """
    id: str
    question: str
    answer: str
    contexts: List[Passage]
    supporting_facts: List[SupportingFact]
    question_type: str = ""   # "bridge" or "comparison"
    level: str = ""           # "easy", "medium", or "hard"

    def get_gold_passages(self) -> List[Passage]:
        """Return only the passages that contain supporting facts"""
        sp_titles = {sf.title for sf in self.supporting_facts}
        return [ctx for ctx in self.contexts if ctx.title in sp_titles]

    def get_distractor_passages(self) -> List[Passage]:
        """Return only the distractor (non-gold) passages"""
        sp_titles = {sf.title for sf in self.supporting_facts}
        return [ctx for ctx in self.contexts if ctx.title not in sp_titles]

    def get_gold_sentences(self) -> List[str]:
        """Return the actual supporting fact sentence strings"""
        title_to_passage = {ctx.title: ctx for ctx in self.contexts}
        sentences = []
        for sf in self.supporting_facts:
            passage = title_to_passage.get(sf.title)
            if passage and 0 <= sf.sentence_index < len(passage.sentences):
                s = passage.sentences[sf.sentence_index]
                if isinstance(s, str):
                    sentences.append(s.strip())
        return sentences

    def format_full_context(self) -> str:
        """Join all context passages into a single str"""
        return " ".join(ctx.title_text for ctx in self.contexts)

    def to_dict(self) -> dict:
        """Convert to simple dict"""
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "contexts": [
                {"title": c.title, "sentences": c.sentences, "passage_id": c.passage_id}
                for c in self.contexts
            ],
            "supporting_facts": [
                {"title": sf.title, "sentence_index": sf.sentence_index}
                for sf in self.supporting_facts
            ],
            "type": self.question_type,
            "level": self.level,
        }


class HotpotQALoader:
    """
    Loads and parses HotpotQA JSON files into standardized examples
    Eg:
        loader = HotpotQALoader("data/hotpot_dev_distractor_v1.json")
        examples = loader.load()
        print(len(examples))
        ex = examples[0]
        print(ex.question, ex.answer)
        print(ex.get_gold_sentences())
    """

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path) # Path to a HotpotQA JSON file lazy-loaded
        self._raw_data: Optional[List[dict]] = None # Raw JSON data lazy-loaded
        self._examples: Optional[List[HotpotQAExample]] = None # Parsed examples lazy-loaded
        self._passage_index: Optional[Dict[str, Passage]] = None # Passage index lazy-built

    def _load_raw(self) -> List[dict]:
        """Lazily load the raw JSON data (cached after first call)"""
        if self._raw_data is None:
            if not self.file_path.exists():
                raise FileNotFoundError(f"File not found: {self.file_path}") 
            with open(self.file_path, "r", encoding="utf-8") as f:
                self._raw_data = json.load(f)
        return self._raw_data

    @staticmethod
    def _parse_context(raw_context: list) -> List[Passage]:
        """
        Parse HotpotQA context into a list of Passage objects
        Raw: ["Scott Derrickson", ["He was born...", "He lives in..."]]
        Parsed: Passage(title="Scott Derrickson", sentences=["He was born...", "He lives in..."])
        """
        passages: List[Passage] = []
        for item in raw_context:
            if isinstance(item, list) and len(item) >= 2:
                title = item[0] if isinstance(item[0], str) else ""
                sentences = item[1] if isinstance(item[1], list) else []
                sentences = [s.strip() for s in sentences if isinstance(s, str)]
                passages.append(Passage(title=title.strip() if title else "", sentences=sentences))
            elif isinstance(item, dict):
                title = item.get("title", "")
                sentences = item.get("sentences", [])
                sentences = [s.strip() for s in sentences if isinstance(s, str)]
                passages.append(Passage(title=title.strip() if isinstance(title, str) else "", sentences=sentences))
            else:
                # Unrecognized shape, skip
                continue
        return passages

    @staticmethod
    def _parse_supporting_facts(raw_sp: list) -> List[SupportingFact]:
        """
        Parse supporting facts into SupportingFact objects
        Raw:  ["Scott Derrickson", 0]
        Parsed: SupportingFact(title="Scott Derrickson", sentence_index=0)
        """
        facts: List[SupportingFact] = []
        for item in raw_sp:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title = item[0] if isinstance(item[0], str) else ""
                idx = int(item[1]) if isinstance(item[1], int) else 0
                facts.append(SupportingFact(title=title.strip(), sentence_index=idx))
            elif isinstance(item, dict):
                title = item.get("title", "")
                idx = int(item.get("sentence_index", 0))
                facts.append(SupportingFact(title=title.strip() if isinstance(title, str) else "", sentence_index=idx))
        return facts

    def _parse_example(self, raw: dict) -> HotpotQAExample:
        """Parse a single raw HotpotQA example into a standardized object"""
        ex_id = str(raw.get("_id", raw.get("id", "")))
        question = raw.get("question", "").strip()
        answer = raw.get("answer", "") or ""
        contexts = self._parse_context(raw.get("context", []))
        supporting_facts = self._parse_supporting_facts(raw.get("supporting_facts", []))
        qtype = raw.get("type", "") or raw.get("question_type", "") or ""
        level = raw.get("level", "") or ""

        return HotpotQAExample(
            id=ex_id,
            question=question,
            answer=answer,
            contexts=contexts,
            supporting_facts=supporting_facts,
            question_type=qtype,
            level=level,
        )

    def load(self, limit: Optional[int] = None) -> List[HotpotQAExample]:
        """
        Load and parse all examples from the file
        limit: If set, only load the first `limit` examples
        examples = loader.load()        # all examples
        examples = loader.load(limit=100)  # first 100 only
        """
        if self._examples is None:
            raw_data = self._load_raw()
            self._examples = [self._parse_example(r) for r in raw_data]

        return self._examples[:limit] if limit is not None else self._examples

    def build_passage_index(self) -> Dict[str, Passage]:
        """
        Build a passage index mapping passage_id => Passage
        passage_id is formatted as "{title}::{paragraph_index}" 
        to handle duplicate titles across different examples. Each unique (title, text) combination gets its own entry
        """
        if self._passage_index is not None:
            return self._passage_index

        examples = self.load()
        seen = set()  # (title, text) dedup
        self._passage_index = {}
        idx = 0

        for ex in examples:
            for ctx in ex.contexts:
                key = (ctx.title, ctx.text)
                if key not in seen:
                    seen.add(key)
                    pid = f"{ctx.title}::{idx}"
                    ctx.passage_id = pid
                    self._passage_index[pid] = ctx
                    idx += 1
        return self._passage_index

    def get_all_passages(self) -> List[Passage]:
        """Get a flat list of all unique passages across all examples"""
        index = self.build_passage_index()
        return list(index.values())

    def get_all_passage_texts(self) -> List[str]:
        """Get all unique passage texts (title + text), for indexing retriever"""
        return [p.title_text for p in self.get_all_passages()]

    def get_stats(self) -> dict:
        """Return summary"""
        examples = self.load()
        types = {}
        levels = {}
        for ex in examples:
            types[ex.question_type] = types.get(ex.question_type, 0) + 1
            levels[ex.level] = levels.get(ex.level, 0) + 1

        return {
            "total_examples": len(examples),
            "question_types": types,
            "difficulty_levels": levels,
            "unique_passages": len(self.build_passage_index()),
            "avg_contexts_per_example": sum(len(ex.contexts) for ex in examples) / max(len(examples), 1),
            "avg_sp_facts_per_example": sum(len(ex.supporting_facts) for ex in examples) / max(len(examples), 1),
        }

    def __len__(self) -> int:
        return len(self.load())

    def __getitem__(self, idx) -> HotpotQAExample:
        return self.load()[idx]

    def __iter__(self):
        return iter(self.load())

    def __repr__(self) -> str:
        return f"HotpotQALoader(file='{self.file_path.name}', loaded={self._examples is not None})"


# --- Loading Functions ---
def load_hotpotqa(file_path: Union[str, Path], limit: Optional[int] = None) -> List[HotpotQAExample]:
    """Load HotpotQA examples."""
    return HotpotQALoader(file_path).load(limit=limit)


def load_hotpotqa_splits(
    data_dir: Union[str, Path],
    splits: Optional[List[str]] = None,
) -> Dict[str, List[HotpotQAExample]]:
    """
    Load multiple HotpotQA splits at once
    data_dir: Directory containing HotpotQA JSON files
    splits: List of filenames to load. Defaults to all available splits
    Returns: Dict mapping split name to list of examples
    """
    data_dir = Path(data_dir)
    if splits is None:
        splits = [
            "hotpot_dev_distractor_v1.json",
            "hotpot_dev_fullwiki_v1.json",
            "hotpot_test_fullwiki_v1.json",
        ]

    result = {}
    for split_file in splits:
        path = data_dir / split_file
        if path.exists():
            name = split_file.replace(".json", "")
            result[name] = HotpotQALoader(path).load()

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.data_loader <path_to_hotpotqa.json> [limit]")
        sys.exit(1)

    file_path = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    loader = HotpotQALoader(file_path)
    examples = loader.load(limit=limit)

    print(f"\nLoaded {len(loader)} total examples from {loader.file_path.name}")
    print(f"Showing first {limit}:\n")

    for ex in examples:
        print(f"  ID:       {ex.id}")
        print(f"  Question: {ex.question}")
        print(f"  Answer:   {ex.answer}")
        print(f"  Type:     {ex.question_type} | Level: {ex.level}")
        print(f"  Contexts: {len(ex.contexts)} passages")
        print(f"  SP Facts: {len(ex.supporting_facts)} facts")
        print(f"  Gold:     {ex.get_gold_sentences()[:2]}...")
        print()

    print("---Stats---")
    for k, v in loader.get_stats().items():
        print(f"  {k}: {v}")


"""
python -m pipeline.data_loader data/hotpot_dev_distractor_v1.json 3

Loaded 7405 total examples from hotpot_dev_distractor_v1.json
Showing first 3:

  ID:       5a8b57f25542995d1e6f1371
  Question: Were Scott Derrickson and Ed Wood of the same nationality?
  Answer:   yes
  Type:     comparison | Level: hard
  Contexts: 10 passages
  SP Facts: 2 facts
  Gold:     ['Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.', 'Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker, actor, writer, producer, and director.']...

  ID:       5a8c7595554299585d9e36b6
  Question: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
  Answer:   Chief of Protocol
  Type:     bridge | Level: hard
  Contexts: 10 passages
  SP Facts: 3 facts
  Gold:     ['Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer.', "Shirley Temple Black (April 23, 1928 – February 10, 2014) was an American actress, singer, dancer, businesswoman, and diplomat who was Hollywood's number one box-office draw as a child actress from 1935 to 1938."]...

  ID:       5a85ea095542994775f606a8
  Question: What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?
  Answer:   Animorphs
  Type:     bridge | Level: hard
  Contexts: 10 passages
  SP Facts: 5 facts
  Gold:     ['The Hork-Bajir Chronicles is the second companion book to the "Animorphs" series, written by K. A. Applegate.', 'With respect to continuity within the series, it takes place before book #23, "The Pretender", although the events told in the story occur between the time of "The Ellimist Chronicles" and "The Andalite Chronicles".']...

---Stats---
  total_examples: 7405
  question_types: {'comparison': 1487, 'bridge': 5918}
  difficulty_levels: {'hard': 7405}
  unique_passages: 66635
  avg_contexts_per_example: 9.952734638757596
  avg_sp_facts_per_example: 2.4314652261985144
"""