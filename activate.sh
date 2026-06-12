#!/bin/bash
# Activate Cosmos3-Nano environment
COSMOS3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$COSMOS3_DIR/.venv/bin/activate"
export COSMOS3_DIR
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_COMPILE_DISABLE=1
export LD_LIBRARY_PATH=
echo "Cosmos3-Nano environment activated"
echo "  Python: $(python --version)"
echo "  Torch:  $(python -c 'import torch; print(torch.__version__)')"
echo "  GPU:    $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
