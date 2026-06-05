#!/bin/bash
# Activate Cosmos3-Nano environment on GB10
source /home/dgx-destro/cosmos3/.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export LD_LIBRARY_PATH=
echo "Cosmos3-Nano environment activated (GB10)"
echo "  Python: $(python --version)"
echo "  Torch:  $(python -c 'import torch; print(torch.__version__)')"
echo "  GPU:    $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
