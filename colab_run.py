#!/usr/bin/env python3
"""
Colab runner — single .py that handles everything.
Upload to Drive, then in a Colab cell:
    %cd /content/drive/MyDrive/hallucination_resistant_multihop_qna-Shreyansh
    !python colab_run.py --limit 10

Or for the full dataset:
    !python colab_run.py
"""

import subprocess, sys, os, time, json, shutil, signal
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────
OLLAMA_MODELS = [
    "qwen3-embedding:8b",   # embeddings (retriever + reranker sentence selection)
    "gemma4:31b",            # generator model_small + model_large + entity extraction
]

PIP_PACKAGES = [
    "sentence-transformers",
    "rapidfuzz",
    "rank_bm25",
    "faiss-cpu",
    "ujson",
]

VERSION = "v39"
# ──────────────────────────────────────────────────────────────────────────


def run(cmd, check=True, capture=False, timeout=600):
    """Run a shell command, print output live unless capturing."""
    print(f"  $ {cmd}")
    if capture:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        if check and r.returncode != 0:
            print(f"STDERR: {r.stderr}")
            raise RuntimeError(f"Command failed: {cmd}")
        return r
    else:
        r = subprocess.run(cmd, shell=True, check=check, timeout=timeout)
        return r


def install_ollama():
    """Install Ollama if not already present."""
    if shutil.which("ollama"):
        r = run("ollama --version", capture=True)
        print(f"  Ollama already installed: {r.stdout.strip()}")
        return
    print("\n[1/5] Installing Ollama...")
    run("curl -fsSL https://ollama.com/install.sh | sh")


def setup_drive_cache():
    """Symlink Ollama model cache to Google Drive for persistence."""
    drive_path = Path("/content/drive/MyDrive/ollama_models")
    local_path = Path.home() / ".ollama" / "models"

    if not Path("/content/drive").exists():
        print("  Google Drive not mounted — skipping cache symlink")
        return

    drive_path.mkdir(parents=True, exist_ok=True)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.is_symlink():
        print(f"  Cache already symlinked → {drive_path}")
        return

    if local_path.exists() and not local_path.is_symlink():
        # Move existing models to Drive first
        run(f'rsync -a "{local_path}/" "{drive_path}/"')
        shutil.rmtree(local_path)

    os.symlink(str(drive_path), str(local_path))
    print(f"  Symlinked {local_path} → {drive_path}")


def start_ollama():
    """Start ollama serve in the background if not already running."""
    r = subprocess.run("curl -sf http://localhost:11434/api/tags", shell=True,
                       capture_output=True, timeout=5)
    if r.returncode == 0:
        print("  Ollama already serving")
        return

    print("\n[2/5] Starting Ollama server...")
    env = os.environ.copy()
    env["OLLAMA_NUM_PARALLEL"] = "3"
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    # Wait for server to be ready
    for i in range(30):
        time.sleep(1)
        r = subprocess.run("curl -sf http://localhost:11434/api/tags", shell=True,
                           capture_output=True, timeout=5)
        if r.returncode == 0:
            print(f"  Ollama ready (took {i+1}s)")
            return
    raise RuntimeError("Ollama failed to start within 30s")


def pull_models():
    """Pull all required Ollama models (skips already-pulled ones)."""
    print("\n[3/5] Pulling models...")
    # Check which models are already pulled
    r = run("curl -s http://localhost:11434/api/tags", capture=True)
    pulled = set()
    try:
        tags = json.loads(r.stdout)
        pulled = {m["name"] for m in tags.get("models", [])}
    except Exception:
        pass

    for model in OLLAMA_MODELS:
        if model in pulled:
            print(f"  ✓ {model} (already pulled)")
        else:
            print(f"  ↓ Pulling {model}...")
            run(f"ollama pull {model}", timeout=1200)

    # Warmup embedding model
    print("  Warming up embedding model...")
    import requests
    requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "qwen3-embedding:8b", "input": "warmup"},
        timeout=300,
    )
    print("  Warmup done")


def install_pip_deps():
    """Install Python dependencies."""
    print("\n[4/5] Installing pip dependencies...")
    run(f"{sys.executable} -m pip install -q {' '.join(PIP_PACKAGES)}")


def detect_gpu():
    """Detect GPU and return info string."""
    try:
        r = run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits",
                capture=True, check=False)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return "unknown"


def run_eval(limit=None):
    """Run the pipeline evaluation."""
    gpu = detect_gpu()
    print(f"\n[5/5] Running evaluation (GPU: {gpu})")
    print(f"  Version: {VERSION}")

    pred_path = f"results/predictions_{VERSION}.json"
    metrics_path = f"results/metrics_{VERSION}.json"
    gold_path = "data/hotpot_dev_distractor_v1.json"

    cmd_parts = [
        sys.executable, "-m", "pipeline.eval",
        "--split", "dev_distractor",
        "--output", pred_path,
        "--eval", gold_path,
        "--metrics", metrics_path,
    ]
    if limit:
        cmd_parts += ["--limit", str(limit)]

    cmd = " ".join(cmd_parts)
    print(f"  $ {cmd}")
    print(f"  {'='*60}")
    start = time.time()
    subprocess.run(cmd_parts, check=False)
    elapsed = time.time() - start
    print(f"  {'='*60}")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Print metrics
    print(f"\n{'='*60}")
    print(f"  RESULTS ({VERSION})")
    print(f"{'='*60}")

    if Path(metrics_path).exists():
        raw = Path(metrics_path).read_text().strip()
        start_idx = raw.find("{")
        if start_idx >= 0:
            try:
                metrics = json.loads(raw[start_idx:])
                for k, v in metrics.items():
                    print(f"  {k:>12}: {v*100:.1f}%")
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=2)
            except json.JSONDecodeError:
                print(f"  (Could not parse metrics, raw output:)")
                print(raw)
        else:
            print(f"  (No JSON in metrics file)")
            print(raw)
    else:
        print(f"  (No metrics file at {metrics_path})")

    # Quick sanity check
    if Path(pred_path).exists():
        with open(pred_path) as f:
            preds = json.load(f)
        print(f"\n  Predictions: {len(preds)} examples")
        for i, (qid, p) in enumerate(preds.items()):
            if i >= 3:
                break
            print(f"    [{qid[:8]}] answer={p['answer']!r}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=f"Colab full pipeline runner ({VERSION})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit examples (default: use config, null=all)")
    parser.add_argument("--skip-setup", action="store_true",
                        help="Skip Ollama install/pull (already done)")
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parent)
    print(f"Working directory: {os.getcwd()}")

    if not args.skip_setup:
        install_ollama()
        setup_drive_cache()
        start_ollama()
        pull_models()
        install_pip_deps()
    else:
        print("Skipping setup (--skip-setup)")
        start_ollama()  # always make sure it's running

    run_eval(limit=args.limit)


if __name__ == "__main__":
    main()
