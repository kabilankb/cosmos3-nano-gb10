#!/bin/bash
# Start vLLM-Omni server for Cosmos3-Nano on GB10
# Supports: text-to-video, image-to-video, forward dynamics, inverse dynamics

MODEL_PATH="/home/dgx-destro/cosmos3/Cosmos3-Nano"
PORT=8000

echo "Starting vLLM-Omni server with Cosmos3-Nano..."
echo "Model: $MODEL_PATH"
echo "Port:  $PORT"
echo ""

docker run --rm \
  --runtime=nvidia \
  --gpus all \
  --ipc=host \
  -p ${PORT}:${PORT} \
  -v ${MODEL_PATH}:/model \
  -v /home/dgx-destro/cosmos3/output:/output \
  vllm/vllm-omni:cosmos3 \
  vllm serve /model \
    --omni \
    --host 0.0.0.0 \
    --port ${PORT} \
    --enable-layerwise-offload \
    --init-timeout 1800
