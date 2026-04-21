"""
vLLM Backend
============
Drop-in replacement for Ollama that talks to a vLLM OpenAI-compatible server.

vLLM exposes an OpenAI-compatible REST API, so we use the same /v1/chat/completions
and /v1/embeddings endpoints that vLLM serves when started with:

    vllm serve <model> --host 0.0.0.0 --port 8000

For embeddings, vLLM also supports a dedicated embedding model:

    vllm serve <embed_model> --host 0.0.0.0 --port 8001 --task embed

Both the Generator and OllamaEmbedder classes detect the backend via config and
delegate to this module instead of making Ollama calls.
"""

from __future__ import annotations

import re
import time
from typing import List, Optional

import numpy as np
import requests

from scripts.logger import get_logger

log = get_logger("vllm_backend")


# ─────────────────────────────────────────────────────────────────────────────
# Text generation
# ─────────────────────────────────────────────────────────────────────────────

class VLLMGeneratorBackend:
    """
    Sends generation requests to a running vLLM server via the
    OpenAI-compatible /v1/chat/completions endpoint.

    Usage
    -----
    backend = VLLMGeneratorBackend(base_url="http://localhost:8000", model="gemma2:9b")
    text = backend.generate(prompt, temperature=0.2)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        request_timeout: int = 1800,
        max_tokens: int = 1024,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.request_timeout = request_timeout
        self.max_tokens = max_tokens
        self.session = requests.Session()

        log.success(f"VLLMGeneratorBackend ready — model={model} at {base_url}")

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        """
        Call the vLLM chat completions endpoint and return the response text.
        Strips <think>...</think> blocks (for reasoning models like DeepSeek-R1).
        """
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            log.step(f"Calling vLLM ({self.model})...")
            t0 = time.time()
            resp = self.session.post(url, json=payload, timeout=self.request_timeout)
            resp.raise_for_status()
            elapsed = time.time() - t0
        except requests.ConnectionError as e:
            raise RuntimeError(
                f"Cannot connect to vLLM at {self.base_url}. "
                f"Is the server running?  Start it with:\n"
                f"  vllm serve {self.model} --host 0.0.0.0 --port 8000"
            ) from e
        except requests.Timeout:
            raise RuntimeError(
                f"vLLM request timed out after {self.request_timeout}s"
            )
        except requests.HTTPError as e:
            raise RuntimeError(f"vLLM HTTP error: {e}") from e

        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as e:
            raise RuntimeError(f"Unexpected vLLM response format: {resp.text[:300]}") from e

        # Strip reasoning model <think> blocks (DeepSeek-R1, Qwen3 thinking mode)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        log.info(f"vLLM returned {len(text)} chars in {elapsed:.1f}s")
        return text

    def health_check(self) -> bool:
        """Return True if vLLM server is reachable and lists at least one model."""
        try:
            resp = self.session.get(f"{self.base_url}/v1/models", timeout=10)
            resp.raise_for_status()
            models = resp.json().get("data", [])
            if models:
                log.info(f"vLLM health check OK — available models: {[m['id'] for m in models]}")
                return True
            log.warning("vLLM server reachable but no models loaded yet.")
            return False
        except Exception as e:
            log.error(f"vLLM health check failed: {e}")
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────────────────────

class VLLMEmbedder:
    """
    Encodes texts using a vLLM embedding server via the OpenAI-compatible
    /v1/embeddings endpoint.

    The embedding server should be started separately from the generator:
        vllm serve Qwen/Qwen3-Embedding-8B \\
            --host 0.0.0.0 --port 8001 --task embed

    This class is a drop-in replacement for OllamaEmbedder — it exposes the
    same .encode() / .encode_query() / .dim interface.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Embedding-8B",
        base_url: str = "http://localhost:8001",
        batch_size: int = 32,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self._dim: Optional[int] = None
        self.session = requests.Session()

        log.info(f"Connecting to vLLM embedder ({model}) at {base_url}...")
        try:
            resp = self.session.get(f"{self.base_url}/v1/models", timeout=10)
            resp.raise_for_status()
            # Probe embedding dimension
            probe = self._embed_batch(["dim_probe"])
            self._dim = probe.shape[1]
            log.success(f"VLLMEmbedder ready — model={model}, dim={self._dim}")
        except requests.ConnectionError as e:
            raise RuntimeError(
                f"Cannot connect to vLLM embedding server at {base_url}. "
                f"Start it with:\n"
                f"  vllm serve {model} --host 0.0.0.0 --port 8001 --task embed"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"VLLMEmbedder init failed: {e}"
            ) from e

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        all_vecs = []
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            if show_progress:
                log.info(f"Embedding batch {i // self.batch_size + 1}/{n_batches} ({len(batch)} texts)...")
            all_vecs.append(self._embed_batch(batch))
        return np.vstack(all_vecs).astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self._embed_batch([text])[0]

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        url = f"{self.base_url}/v1/embeddings"
        payload = {"model": self.model, "input": texts}

        try:
            resp = self.session.post(url, json=payload, timeout=1800)
            resp.raise_for_status()
        except requests.ConnectionError as e:
            raise RuntimeError(f"vLLM embedding server not reachable at {url}") from e
        except requests.HTTPError as e:
            raise RuntimeError(f"vLLM /v1/embeddings returned error: {e}") from e

        try:
            data = resp.json()
            # OpenAI format: {"data": [{"embedding": [...], "index": 0}, ...]}
            sorted_items = sorted(data["data"], key=lambda x: x["index"])
            vecs = np.array([item["embedding"] for item in sorted_items], dtype=np.float32)
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"Unexpected vLLM embeddings response: {resp.text[:300]}") from e

        return self._l2_normalize(vecs)

    @staticmethod
    def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vecs / norms

    def health_check(self) -> bool:
        try:
            resp = self.session.get(f"{self.base_url}/v1/models", timeout=10)
            resp.raise_for_status()
            return True
        except Exception:
            return False
