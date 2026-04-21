"""
test_vllm_integration.py
========================
Tests the vLLM integration WITHOUT needing a real GPU or model.

How it works:
  1. Spins up two tiny Flask servers on ports 18000 and 18001 that mimic
     the vLLM OpenAI-compatible API (/v1/models, /v1/chat/completions,
     /v1/embeddings).
  2. Runs all vLLM integration code against those mock servers.
  3. Tests config loading, Generator routing, Embedder, and health checks.

Run with:
    python tests/test_vllm_integration.py
    
Expected output: all tests PASS, no server or GPU required.
"""

import json
import os
import sys
import threading
import time
import unittest

import numpy as np
from flask import Flask, jsonify, request

# ── make sure project root is on sys.path ─────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# Mock vLLM servers
# ─────────────────────────────────────────────────────────────────────────────

GENERATOR_PORT = 18000
EMBED_PORT     = 18001
EMBED_DIM      = 16   # tiny dimension — just enough to test shape/dtype

# Canned generator response — valid JSON that our parser should handle
MOCK_ANSWER_JSON = json.dumps({
    "reasoning": "Source 1 says X. Source 2 says Y. Therefore Z.",
    "answer": "1999",
    "supporting_fact_numbers": [0, 1, 5, 6]
})


def make_generator_app():
    app = Flask("mock_generator")
    app.logger.disabled = True

    @app.route("/v1/models")
    def models():
        return jsonify({"data": [{"id": "mock-generator", "object": "model"}]})

    @app.route("/v1/chat/completions", methods=["POST"])
    def chat():
        # Echo back a valid pipeline-parseable response
        return jsonify({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": MOCK_ANSWER_JSON
                },
                "finish_reason": "stop"
            }],
            "model": "mock-generator",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}
        })

    return app


def make_embed_app():
    app = Flask("mock_embedder")
    app.logger.disabled = True

    @app.route("/v1/models")
    def models():
        return jsonify({"data": [{"id": "mock-embedder", "object": "model"}]})

    @app.route("/v1/embeddings", methods=["POST"])
    def embeddings():
        data = request.get_json()
        texts = data.get("input", [])
        if isinstance(texts, str):
            texts = [texts]
        # Return random unit vectors of fixed dimension
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((len(texts), EMBED_DIM)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms
        return jsonify({
            "data": [
                {"embedding": vecs[i].tolist(), "index": i, "object": "embedding"}
                for i in range(len(texts))
            ],
            "model": "mock-embedder",
        })

    return app


def start_server(app, port):
    """Start a Flask app in a daemon thread."""
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    t = threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=port, use_reloader=False),
        daemon=True
    )
    t.start()
    # Wait until server is accepting connections
    import socket
    for _ in range(30):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                return t
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"Mock server on port {port} did not start in time")


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestVLLMBackend(unittest.TestCase):

    # ── VLLMGeneratorBackend ─────────────────────────────────────────────────

    def test_generator_health_check(self):
        from pipeline.vllm_backend import VLLMGeneratorBackend
        b = VLLMGeneratorBackend(
            base_url=f"http://127.0.0.1:{GENERATOR_PORT}",
            model="mock-generator"
        )
        self.assertTrue(b.health_check(), "Generator health check should return True")

    def test_generator_returns_string(self):
        from pipeline.vllm_backend import VLLMGeneratorBackend
        b = VLLMGeneratorBackend(
            base_url=f"http://127.0.0.1:{GENERATOR_PORT}",
            model="mock-generator"
        )
        result = b.generate("What year did Guns N' Roses release Oh My God?")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_generator_response_is_parseable_json(self):
        from pipeline.vllm_backend import VLLMGeneratorBackend
        b = VLLMGeneratorBackend(
            base_url=f"http://127.0.0.1:{GENERATOR_PORT}",
            model="mock-generator"
        )
        raw = b.generate("test prompt")
        parsed = json.loads(raw)
        self.assertIn("answer", parsed)
        self.assertIn("supporting_fact_numbers", parsed)
        self.assertEqual(parsed["answer"], "1999")

    def test_generator_strips_think_blocks(self):
        """VLLMGeneratorBackend should strip <think>...</think> from responses."""
        from pipeline.vllm_backend import VLLMGeneratorBackend
        import unittest.mock as mock
        b = VLLMGeneratorBackend(
            base_url=f"http://127.0.0.1:{GENERATOR_PORT}",
            model="mock-generator"
        )
        # Patch the session to return a response with a <think> block
        raw_with_think = '<think>Let me reason step by step...</think>\n{"answer": "1999", "supporting_fact_numbers": [0]}'
        fake_response = mock.MagicMock()
        fake_response.json.return_value = {
            "choices": [{"message": {"content": raw_with_think}}]
        }
        fake_response.raise_for_status = mock.MagicMock()
        with mock.patch.object(b.session, "post", return_value=fake_response):
            result = b.generate("test")
        self.assertNotIn("<think>", result)
        self.assertIn('"answer"', result)

    def test_generator_connection_error(self):
        """Should raise RuntimeError with helpful message when server is down."""
        from pipeline.vllm_backend import VLLMGeneratorBackend
        b = VLLMGeneratorBackend(
            base_url="http://127.0.0.1:19999",  # nothing running here
            model="mock-generator",
            request_timeout=2
        )
        with self.assertRaises(RuntimeError) as ctx:
            b.generate("test")
        self.assertIn("vllm serve", str(ctx.exception).lower() or str(ctx.exception))

    # ── VLLMEmbedder ─────────────────────────────────────────────────────────

    def test_embedder_health_check(self):
        from pipeline.vllm_backend import VLLMEmbedder
        e = VLLMEmbedder(
            model="mock-embedder",
            base_url=f"http://127.0.0.1:{EMBED_PORT}"
        )
        self.assertTrue(e.health_check())

    def test_embedder_dim(self):
        from pipeline.vllm_backend import VLLMEmbedder
        e = VLLMEmbedder(
            model="mock-embedder",
            base_url=f"http://127.0.0.1:{EMBED_PORT}"
        )
        self.assertEqual(e.dim, EMBED_DIM)

    def test_embedder_encode_shape(self):
        from pipeline.vllm_backend import VLLMEmbedder
        e = VLLMEmbedder(
            model="mock-embedder",
            base_url=f"http://127.0.0.1:{EMBED_PORT}"
        )
        texts = ["Hello world", "Multi-hop question", "Another passage"]
        vecs = e.encode(texts)
        self.assertEqual(vecs.shape, (3, EMBED_DIM))
        self.assertEqual(vecs.dtype, np.float32)

    def test_embedder_encode_query_shape(self):
        from pipeline.vllm_backend import VLLMEmbedder
        e = VLLMEmbedder(
            model="mock-embedder",
            base_url=f"http://127.0.0.1:{EMBED_PORT}"
        )
        vec = e.encode_query("What year did X happen?")
        self.assertEqual(vec.shape, (EMBED_DIM,))

    def test_embedder_vectors_are_unit_normalized(self):
        from pipeline.vllm_backend import VLLMEmbedder
        e = VLLMEmbedder(
            model="mock-embedder",
            base_url=f"http://127.0.0.1:{EMBED_PORT}"
        )
        vecs = e.encode(["test sentence one", "test sentence two"])
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, np.ones(2), atol=1e-5,
                                   err_msg="Embeddings should be L2-normalized")

    def test_embedder_batching(self):
        """encode() should handle more texts than batch_size without errors."""
        from pipeline.vllm_backend import VLLMEmbedder
        e = VLLMEmbedder(
            model="mock-embedder",
            base_url=f"http://127.0.0.1:{EMBED_PORT}",
            batch_size=3
        )
        texts = [f"Sentence number {i}" for i in range(10)]
        vecs = e.encode(texts)
        self.assertEqual(vecs.shape, (10, EMBED_DIM))

    def test_embedder_connection_error(self):
        from pipeline.vllm_backend import VLLMEmbedder
        with self.assertRaises(RuntimeError) as ctx:
            VLLMEmbedder(
                model="mock-embedder",
                base_url="http://127.0.0.1:19998"  # nothing running
            )
        self.assertIn("vllm serve", str(ctx.exception))


class TestGeneratorRouting(unittest.TestCase):
    """Test that Generator correctly routes to vLLM vs Ollama based on use_vllm flag."""

    def test_ollama_mode_no_vllm_backends(self):
        from pipeline.generator import Generator
        gen = Generator(use_vllm=False)
        self.assertFalse(gen.use_vllm)
        self.assertIsNone(gen._vllm_small)
        self.assertIsNone(gen._vllm_large)

    def test_vllm_mode_creates_backends(self):
        from pipeline.generator import Generator
        gen = Generator(
            use_vllm=True,
            vllm_base_url=f"http://127.0.0.1:{GENERATOR_PORT}",
            vllm_model_small="mock-generator",
            vllm_model_large="mock-generator",
        )
        self.assertTrue(gen.use_vllm)
        self.assertIsNotNone(gen._vllm_small)
        self.assertIsNotNone(gen._vllm_large)

    def test_vllm_same_model_shares_backend_instance(self):
        """When small == large model, only one backend should be created."""
        from pipeline.generator import Generator
        gen = Generator(
            use_vllm=True,
            vllm_base_url=f"http://127.0.0.1:{GENERATOR_PORT}",
            vllm_model_small="mock-generator",
            vllm_model_large="mock-generator",
        )
        self.assertIs(gen._vllm_small, gen._vllm_large,
                      "Same model should share one backend instance")

    def test_call_llm_routes_to_vllm(self):
        """_call_llm should call _call_vllm when use_vllm=True."""
        from pipeline.generator import Generator
        import unittest.mock as mock
        gen = Generator(
            use_vllm=True,
            vllm_base_url=f"http://127.0.0.1:{GENERATOR_PORT}",
            vllm_model_small="mock-generator",
            vllm_model_large="mock-generator",
        )
        with mock.patch.object(gen, "_call_vllm", return_value="vllm_response") as mv:
            result = gen._call_llm("prompt", "mock-generator", 0.2)
        mv.assert_called_once_with("prompt", "mock-generator", 0.2)
        self.assertEqual(result, "vllm_response")

    def test_call_llm_routes_to_ollama(self):
        """_call_llm should call _call_ollama when use_vllm=False."""
        from pipeline.generator import Generator
        import unittest.mock as mock
        gen = Generator(use_vllm=False)
        with mock.patch.object(gen, "_call_ollama", return_value="ollama_response") as mo:
            result = gen._call_llm("prompt", "some-model", 0.1)
        mo.assert_called_once_with("prompt", "some-model", 0.1)
        self.assertEqual(result, "ollama_response")

    def test_no_infinite_recursion_in_call_llm(self):
        """_call_llm must not call itself (regression test for the bug we fixed)."""
        from pipeline.generator import Generator
        import unittest.mock as mock
        gen = Generator(use_vllm=False)
        call_count = [0]
        original = gen._call_ollama
        def counting_ollama(prompt, model, temp):
            call_count[0] += 1
            if call_count[0] > 5:
                raise RecursionError("_call_llm is recursing into itself!")
            return "ok"
        gen._call_ollama = counting_ollama
        result = gen._call_llm("p", "m", 0.1)
        self.assertEqual(call_count[0], 1, "_call_ollama should be called exactly once")


class TestBuildEmbedderFactory(unittest.TestCase):
    """Test that build_embedder() returns the right class based on config."""

    def test_returns_ollama_embedder_by_default(self):
        from pipeline.embedder import OllamaEmbedder, build_embedder
        import unittest.mock as mock
        # Patch Ollama __init__ so no real server is needed
        with mock.patch("pipeline.embedder.OllamaEmbedder.__init__", return_value=None):
            result = build_embedder(None)
        self.assertIsInstance(result, OllamaEmbedder)

    def test_returns_vllm_embedder_when_configured(self):
        from pipeline.embedder import build_embedder
        from pipeline.vllm_backend import VLLMEmbedder
        from unittest.mock import MagicMock
        # Build a minimal mock config with use_vllm=True
        cfg = MagicMock()
        cfg.retriever.use_vllm = True
        cfg.retriever.vllm_embed_model = "mock-embedder"
        cfg.retriever.vllm_embed_base_url = f"http://127.0.0.1:{EMBED_PORT}"
        cfg.retriever.batch_size = 8
        result = build_embedder(cfg)
        self.assertIsInstance(result, VLLMEmbedder)

    def test_returns_ollama_embedder_when_use_vllm_false(self):
        from pipeline.embedder import OllamaEmbedder, build_embedder
        from unittest.mock import MagicMock, patch
        cfg = MagicMock()
        cfg.retriever.use_vllm = False
        cfg.retriever.embed_model = "nomic-embed-text"
        cfg.retriever.ollama_base_url = "http://localhost:11434"
        cfg.retriever.batch_size = 32
        # Patch Ollama __init__ so no real server is needed
        with patch("pipeline.embedder.OllamaEmbedder.__init__", return_value=None):
            result = build_embedder(cfg)
        self.assertIsInstance(result, OllamaEmbedder)


class TestConfigLoading(unittest.TestCase):
    """Test that configs/vllm.yaml loads correctly."""

    def test_vllm_yaml_loads(self):
        from scripts.config import load_config
        cfg = load_config(os.path.join(ROOT, "configs", "vllm.yaml"))
        self.assertTrue(cfg.generator.use_vllm)
        self.assertTrue(cfg.retriever.use_vllm)

    def test_vllm_yaml_generator_fields(self):
        from scripts.config import load_config
        cfg = load_config(os.path.join(ROOT, "configs", "vllm.yaml"))
        g = cfg.generator
        self.assertEqual(g.vllm_base_url, "http://localhost:8000")
        self.assertEqual(g.vllm_model_small, "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(g.vllm_model_large, "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(g.vllm_max_tokens, 1024)
        self.assertEqual(g.request_timeout, 300)

    def test_vllm_yaml_retriever_fields(self):
        from scripts.config import load_config
        cfg = load_config(os.path.join(ROOT, "configs", "vllm.yaml"))
        r = cfg.retriever
        self.assertEqual(r.vllm_embed_model, "Qwen/Qwen3-Embedding-8B")
        self.assertEqual(r.vllm_embed_base_url, "http://localhost:8001")
        self.assertEqual(r.batch_size, 64)

    def test_vllm_yaml_parallel_workers(self):
        from scripts.config import load_config
        cfg = load_config(os.path.join(ROOT, "configs", "vllm.yaml"))
        self.assertEqual(cfg.eval.parallel_workers, 8)

    def test_default_yaml_still_has_no_vllm(self):
        """default.yaml should still work and default use_vllm to False."""
        from scripts.config import load_config
        cfg = load_config(os.path.join(ROOT, "configs", "default.yaml"))
        self.assertFalse(cfg.generator.use_vllm)
        self.assertFalse(cfg.retriever.use_vllm)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Starting mock vLLM servers...")
    start_server(make_generator_app(), GENERATOR_PORT)
    start_server(make_embed_app(), EMBED_PORT)
    print(f"  ✓ Mock generator  → http://127.0.0.1:{GENERATOR_PORT}")
    print(f"  ✓ Mock embedder   → http://127.0.0.1:{EMBED_PORT}")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestVLLMBackend,
        TestGeneratorRouting,
        TestBuildEmbedderFactory,
        TestConfigLoading,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
