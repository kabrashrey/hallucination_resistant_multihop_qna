#!/bin/bash
# =============================================================================
# CARC SLURM Job Script — HotpotQA vLLM Pipeline
# =============================================================================
# Submits a job that:
#   1. Starts two vLLM servers (generator + embeddings) on the allocated node
#   2. Waits for both to be healthy
#   3. Runs the full pipeline evaluation with configs/vllm.yaml
#   4. Saves predictions + metrics to results/
#
# Usage:
#   sbatch carc_run.slurm
#
# Tune #SBATCH directives below to your CARC allocation and model sizes.
# Check available partitions with: sinfo
# Check GPU availability with:     squeue -p gpu
# =============================================================================

#SBATCH --job-name=hotpot-vllm
#SBATCH --partition=gpu                 # Change to your GPU partition name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1              # Request 1 A100; use gpu:a100:2 for tensor parallel
#SBATCH --time=12:00:00                # 12 hr wall time — full 7,405-example run
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@usc.edu  # ← update this

# =============================================================================
# CONFIG — edit these to match your CARC environment
# =============================================================================
GENERATOR_MODEL="Qwen/Qwen2.5-7B-Instruct"
EMBED_MODEL="Qwen/Qwen3-Embedding-8B"
GENERATOR_PORT=8000
EMBED_PORT=8001

VLLM_ENV="vllm"          # conda env with vllm installed
PIPELINE_ENV="hotpot"    # conda env with pipeline deps (sentence-transformers, faiss, etc.)
PROJECT_DIR="$HOME/hallucination_resistant_multihop_qna"
CONFIG="configs/vllm.yaml"
VERSION="v40"            # bump this each new run to avoid overwriting old results
# =============================================================================

set -euo pipefail
mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

echo "============================================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $(hostname)"
echo "Start time : $(date)"
echo "GPU(s)     : $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',')"
echo "============================================================"

# Load modules (adjust to CARC's module system)
module purge
module load gcc/11.3.0
module load cuda/12.1.0
module load conda

# ── 1. Start vLLM generator server ───────────────────────────────────────────
echo "[1/4] Starting vLLM generator server on port $GENERATOR_PORT..."
conda run -n "$VLLM_ENV" \
    vllm serve "$GENERATOR_MODEL" \
    --host 0.0.0.0 \
    --port "$GENERATOR_PORT" \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --dtype auto \
    --gpu-memory-utilization 0.45 \
    > logs/vllm_generator_${SLURM_JOB_ID}.log 2>&1 &
GENERATOR_PID=$!
echo "  Generator PID: $GENERATOR_PID"

# ── 2. Start vLLM embedding server ───────────────────────────────────────────
echo "[2/4] Starting vLLM embedding server on port $EMBED_PORT..."
conda run -n "$VLLM_ENV" \
    vllm serve "$EMBED_MODEL" \
    --host 0.0.0.0 \
    --port "$EMBED_PORT" \
    --task embed \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.45 \
    > logs/vllm_embed_${SLURM_JOB_ID}.log 2>&1 &
EMBED_PID=$!
echo "  Embedding PID: $EMBED_PID"

# ── 3. Wait for both servers to be healthy ────────────────────────────────────
echo "[3/4] Waiting for vLLM servers to be ready..."

wait_for_server() {
    local port=$1
    local name=$2
    local max_wait=180   # 3 minutes max
    local elapsed=0
    echo "  Waiting for $name on port $port..."
    until curl -sf "http://localhost:$port/v1/models" > /dev/null 2>&1; do
        if [ $elapsed -ge $max_wait ]; then
            echo "  ERROR: $name failed to start within ${max_wait}s"
            echo "  Check logs/vllm_${name}_${SLURM_JOB_ID}.log for details"
            kill $GENERATOR_PID $EMBED_PID 2>/dev/null || true
            exit 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        echo "  ...still waiting (${elapsed}s elapsed)"
    done
    echo "  ✓ $name ready on port $port"
}

wait_for_server "$GENERATOR_PORT" "generator"
wait_for_server "$EMBED_PORT"     "embed"
echo "Both vLLM servers are healthy."

# ── 4. Run the pipeline ───────────────────────────────────────────────────────
echo "[4/4] Running pipeline evaluation..."
echo "  Config  : $CONFIG"
echo "  Version : $VERSION"
echo "  Time    : $(date)"

conda run -n "$PIPELINE_ENV" \
    python pipeline/eval.py \
        --config "$CONFIG" \
        --version "$VERSION" \
        --checkpoint-interval 25

PIPELINE_EXIT=$?

# ── Cleanup ───────────────────────────────────────────────────────────────────
echo "Stopping vLLM servers..."
kill $GENERATOR_PID $EMBED_PID 2>/dev/null || true
wait $GENERATOR_PID $EMBED_PID 2>/dev/null || true

echo "============================================================"
echo "End time   : $(date)"
if [ $PIPELINE_EXIT -eq 0 ]; then
    echo "Status     : SUCCESS"
    echo "Results    : results/predictions_${VERSION}.json"
    echo "            results/metrics_${VERSION}.json"
else
    echo "Status     : FAILED (exit code $PIPELINE_EXIT)"
fi
echo "============================================================"
exit $PIPELINE_EXIT
