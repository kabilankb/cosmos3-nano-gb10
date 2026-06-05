#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
COSMOS3_DIR="$(pwd)"

PLATFORM="${1:-gb10}"
MODE="${2:-interactive}"

case "$PLATFORM" in
    gb10)           IMAGE="cosmos3-nano:gb10" ;;
    jetson-thor|thor) IMAGE="cosmos3-nano:jetson-thor" ;;
    *)
        echo "Usage: $0 [gb10|jetson-thor] [interactive|generate|trajectory|t2v]"
        exit 1
        ;;
esac

COMMON_ARGS=(
    --rm
    --runtime=nvidia
    --gpus all
    --ipc=host
    -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    -e TORCH_COMPILE_DISABLE=1
    -e LD_LIBRARY_PATH=""
    -e HF_HUB_ENABLE_HF_TRANSFER=1
    -v "$COSMOS3_DIR/Cosmos3-Nano:/workspace/cosmos3/Cosmos3-Nano:ro"
    -v "$COSMOS3_DIR/Cosmos3-Nano-assets:/workspace/cosmos3/Cosmos3-Nano-assets:ro"
    -v "$COSMOS3_DIR/output:/workspace/cosmos3/output"
    -v "$COSMOS3_DIR/inputs:/workspace/cosmos3/inputs"
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface"
)

case "$MODE" in
    interactive|bash|shell)
        echo "Starting interactive shell ($PLATFORM)..."
        docker run -it "${COMMON_ARGS[@]}" "$IMAGE"
        ;;
    generate|i2v)
        echo "Starting image-to-video generator ($PLATFORM)..."
        docker run -it "${COMMON_ARGS[@]}" "$IMAGE" \
            bash -c "source .venv/bin/activate && python generate_video.py"
        ;;
    trajectory|inverse)
        echo "Starting trajectory extraction ($PLATFORM)..."
        docker run -it "${COMMON_ARGS[@]}" "$IMAGE" \
            bash -c "source .venv/bin/activate && python extract_trajectory.py"
        ;;
    t2v|text2video)
        echo "Running text-to-video test ($PLATFORM)..."
        docker run "${COMMON_ARGS[@]}" "$IMAGE" \
            bash -c "source .venv/bin/activate && python test_t2v.py"
        ;;
    *)
        echo "Usage: $0 [gb10|jetson-thor] [interactive|generate|trajectory|t2v]"
        exit 1
        ;;
esac
