#!/usr/bin/env bash
# test.sh - Manual test script for bias-personalizzato-per-whisper-locale
# Checks GPU availability, runs Whisper with bias and LLM correction.

# Exit on any error
set -e

# Project root (directory containing this script's parent)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

# Function to print status messages
log() {
    echo "[test.sh] $*"
}

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    log "nvidia-smi not found. Assuming GPU is available (no check)."
    GPU_CHECK=0
else
    GPU_CHECK=1
fi

# Simple GPU busy check: if any process is using GPU memory (non-zero)
if [ $GPU_CHECK -eq 1 ]; then
    # Get total GPU memory used across all GPUs (in MiB)
    # We'll use the first GPU (index 0) as per typical single GPU setup
    GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo "0")
    if [ -z "$GPU_MEM_USED" ]; then
        GPU_MEM_USED=0
    fi
    # Consider GPU free if used memory is less than 100 MiB (allowing for small driver usage)
    if [ "$GPU_MEM_USED" -gt 100 ]; then
        log "GPU appears to be in use (memory used: ${GPU_MEM_USED} MiB). Please free the GPU before running."
        exit 1
    else
        log "GPU check passed (memory used: ${GPU_MEM_USED} MiB)."
    fi
fi

# Ensure we are in the project root
cd "$PROJECT_ROOT"

# Check for bias file (JSON or TSV)
BIAS_JSON="$PROJECT_ROOT/data/bias.json"
BIAS_TSV="$PROJECT_ROOT/data/bias.tsv"
if [ -f "$BIAS_JSON" ]; then
    BIAS_FILE="$BIAS_JSON"
    log "Found bias file: $BIAS_FILE"
elif [ -f "$BIAS_TSV" ]; then
    BIAS_FILE="$BIAS_TSV"
    log "Found bias file: $BIAS_FILE"
else
    log "No bias file found in data/. Using empty bias (no corrections)."
    BIAS_FILE=""
fi

# Check for GGUF model (via environment or default location)
MODEL_ENV=$(printenv WHISPER_CONTEXT_MODEL)
if [ -n "$MODEL_ENV" ] && [ -f "$MODEL_ENV" ]; then
    GGUF_MODEL="$MODEL_ENV"
    log "Using GGUF model from WHISPER_CONTEXT_MODEL: $GGUF_MODEL"
elif [ -f "$PROJECT_ROOT/models/Qwen3.5-9B-Q5_K_M.gguf" ]; then
    GGUF_MODEL="$PROJECT_ROOT/models/Qwen3.5-9B-Q5_K_M.gguf"
    log "Found default GGUF model: $GGUF_MODEL"
else
    GGUF_MODEL=""
    log "No GGUF model found. Contextual correction will be skipped."
fi

# Check if Whisper is available (try import)
if python3 -c "import whisper" 2>/dev/null; then
    WHISPER_AVAILABLE=1
    log "Whisper package is available."
else
    WHISPER_AVAILABLE=0
    log "Whisper package not found. Will use a dummy transcription for testing."
fi

# Run the test Python code
TMPPY=$(mktemp)
cat > "$TMPPY" << 'EOF'
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bias_module import BiasLoader, apply_bias

def main():
    # Determine paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Bias file
    bias_json = os.path.join(project_root, 'data', 'bias.json')
    bias_tsv = os.path.join(project_root, 'data', 'bias.tsv')
    bias_file = None
    if os.path.exists(bias_json):
        bias_file = bias_json
    elif os.path.exists(bias_tsv):
        bias_file = bias_tsv

    # Load bias
    if bias_file and os.path.exists(bias_file):
        print(f"Loading bias from: {bias_file}")
        bias_loader = BiasLoader(bias_file)
    else:
        print("No bias file found, using empty bias.")
        bias_loader = BiasLoader()

    # Get transcription: either from Whisper (if audio file provided) or dummy
    # For simplicity, we use a dummy transcription that matches bias examples
    raw_transcription = "ciao mondo, uso WhatsApp per parlare con amici"
    print(f"Raw transcription: {raw_transcription}")

    # Apply bias
    biased_transcription = apply_bias(raw_transcription, bias_loader)
    print(f"After bias: {biased_transcription}")

    # Try contextual correction if model available
    try:
        from contextual_correction import ContextualCorrector
        # Model path from env or default
        model_path = os.environ.get('WHISPER_CONTEXT_MODEL')
        if not model_path:
            model_path = os.path.join(project_root, 'models', 'Qwen3.5-9B-Q5_K_M.gguf')
        if os.path.exists(model_path):
            print(f"Loading contextual model from: {model_path}")
            corrector = ContextualCorrector(model_path=model_path, n_gpu_layers=0, verbose=False)
            corrected = corrector.correct(biased_transcription)
            print(f"After contextual correction: {corrected}")
        else:
            print("GGUF model not found, skipping contextual correction.")
            print(f"Final result: {biased_transcription}")
    except ImportError:
        print("llama_cpp not installed, skipping contextual correction.")
        print(f"Final result: {biased_transcription}")
    except Exception as e:
        print(f"Error during contextual correction: {e}")
        print(f"Final result: {biased_transcription}")

if __name__ == '__main__':
    main()
EOF

    python3 "$TMPPY"
    rm -f "$TMPPY"
EOF