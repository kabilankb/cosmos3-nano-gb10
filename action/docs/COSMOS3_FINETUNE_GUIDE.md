# Cosmos3-Nano Fine-Tuning Guide — LoRA on RTX PRO 6000

## Overview

Fine-tune Cosmos3-Nano (16B) with LoRA adapters on your SO-101 robot data for more accurate action trajectories and video generation.

| Setting | Value |
|---|---|
| **Base model** | Cosmos3-Nano (16B) |
| **Method** | LoRA (rank 16, alpha 32) |
| **Target GPU** | RTX PRO 6000 (96GB) or 2x RTX 6000 Ada (48GB) |
| **Training data** | LeRobot SO-101 dataset (JSONL format) |
| **Trainable params** | ~256 MB (LoRA adapters only) |
| **Full model frozen** | ~32 GB in BF16 |

---

## Prerequisites

```bash
# System
- NVIDIA GPU: RTX PRO 6000 (96GB) or equivalent
- CUDA 13.0+ (or 12.8)
- Ubuntu 22.04+ (x86_64)
- ~200GB free disk space

# Software (already installed on GB10, replicate on training machine)
- Python 3.13
- PyTorch 2.12+cu130
- cosmos-framework
```

---

## Step 1 — Prepare Environment (on training machine)

```bash
# Clone and install cosmos-framework
git clone https://github.com/NVIDIA/cosmos-framework.git
cd cosmos-framework

# Create venv
uv venv --python 3.13 --seed .venv
source .venv/bin/activate

# Install with training dependencies (CUDA 13.0)
uv sync --all-extras --group=cu130-train
export LD_LIBRARY_PATH=

# Or with Docker (recommended)
docker pull nvcr.io/nvidia/pytorch:25.09-py3
docker run -it --runtime=nvidia --ipc=host \
  -v .:/workspace -v /workspace/.venv \
  -v /root/.cache:/root/.cache \
  -e HF_TOKEN="$HF_TOKEN" \
  $(docker build -q .)
```

---

## Step 2 — Prepare Your SO-101 Dataset

### Option A: Use existing LeRobot dataset

```bash
# Download sample SO-101 dataset
uvx hf@latest download --repo-type dataset \
    sreetz-nv/so101_teleop_vials_rack_left \
    --local-dir examples/data/so101_vials
```

### Option B: Record your own with LeRobot

```bash
pip install lerobot

# Record teleoperation episodes
python -m lerobot.scripts.control_robot \
    --robot.type=so101_follower \
    --control.type=teleoperate \
    --control.fps=30 \
    --control.dataset_repo_id=your-user/my_so101_binpick \
    --control.num_episodes=50
```

### Convert to JSONL format for cosmos-framework

The framework expects a JSONL dataset with video paths and captions.

#### 2a. Extract videos from LeRobot dataset

```python
"""Convert LeRobot parquet+video dataset to cosmos-framework JSONL format."""
import json
import os
import pandas as pd
from pathlib import Path

LEROBOT_DIR = "examples/data/so101_vials"
OUTPUT_DIR = "examples/data/so101_sft_dataset"
os.makedirs(f"{OUTPUT_DIR}/train/videos", exist_ok=True)

info = json.load(open(f"{LEROBOT_DIR}/meta/info.json"))
num_episodes = info["total_episodes"]
fps = info["fps"]

jsonl_entries = []

for ep_idx in range(num_episodes):
    # Video path (ego camera)
    video_src = f"{LEROBOT_DIR}/videos/observation.images.ego/chunk-000/file-{ep_idx:03d}.mp4"
    video_dst = f"videos/episode_{ep_idx:06d}.mp4"

    if os.path.exists(video_src):
        os.symlink(os.path.abspath(video_src),
                   f"{OUTPUT_DIR}/train/{video_dst}")

    # Read episode parquet for frame count
    parquet_path = f"{LEROBOT_DIR}/data/chunk-000/file-{ep_idx:03d}.parquet"
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        num_frames = len(df)
        duration = num_frames / fps

        entry = {
            "uuid": f"so101_ep{ep_idx:06d}",
            "duration": round(duration, 2),
            "width": 640,
            "height": 480,
            "vision_path": video_dst,
            "t2w_windows": [{
                "start_frame": 0,
                "end_frame": num_frames - 1,
                "temporal_interval": 1,
                "caption": f"A SO-101 robot arm performs a pick and place task with vials on a rack. "
                           f"The robot moves smoothly through {num_frames} frames at {fps} fps."
            }]
        }
        jsonl_entries.append(entry)

with open(f"{OUTPUT_DIR}/train/video_dataset_file.jsonl", "w") as f:
    for entry in jsonl_entries:
        f.write(json.dumps(entry) + "\n")

print(f"Created {len(jsonl_entries)} entries")
```

#### 2b. Generate better captions with VLM (optional, recommended)

```bash
# Start a captioning server
uvx --with nvidia-cuda-runtime-cu12 \
    vllm@0.19.0 serve Qwen/Qwen3-VL-8B-Instruct-FP8 \
    --tensor-parallel-size 1 \
    --allowed-local-media-path / \
    --port 8001

# Caption your videos
python -m cosmos_framework.scripts.caption_from_video \
    --video examples/data/so101_sft_dataset/train/videos/ \
    -o examples/data/so101_sft_dataset/train/captions \
    --server http://localhost:8001/v1

# Create JSONL from captions
python -m cosmos_framework.scripts.captions_to_sft_jsonl \
    --captions-dir examples/data/so101_sft_dataset/train/captions \
    --videos-dir examples/data/so101_sft_dataset/train/videos \
    -o examples/data/so101_sft_dataset/train/video_dataset_file.jsonl
```

### JSONL Entry Format

Each line in `video_dataset_file.jsonl`:

```json
{
    "uuid": "so101_ep000015",
    "duration": 8.1,
    "width": 640,
    "height": 480,
    "vision_path": "videos/episode_000015.mp4",
    "t2w_windows": [
        {
            "start_frame": 0,
            "end_frame": 242,
            "temporal_interval": 1,
            "caption": "A SO-101 robot arm reaches toward a rack of vials. The shoulder rotates left while the elbow extends downward. The gripper opens wide, approaches a blue vial, then closes firmly around it. The arm lifts the vial and places it in a new position on the rack."
        }
    ]
}
```

### Dataset Directory Structure

```
examples/data/so101_sft_dataset/
└── train/
    ├── video_dataset_file.jsonl     # One JSON per line
    └── videos/
        ├── episode_000000.mp4
        ├── episode_000001.mp4
        └── ...
```

---

## Step 3 — Download and Convert Base Checkpoint

```bash
# Download Wan2.2 VAE (required for video tokenization)
uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
    --local-dir examples/checkpoints/wan22_vae --quiet

# Convert Cosmos3-Nano from HuggingFace safetensors to DCP format
python -m cosmos_framework.scripts.convert_model_to_dcp \
    -o examples/checkpoints/Cosmos3-Nano \
    --checkpoint-path Cosmos3-Nano
```

If you already downloaded the model on GB10, copy it to the training machine:

```bash
# From GB10:
rsync -avP ~/cosmos3/Cosmos3-Nano/ training-machine:/path/to/cosmos-framework/Cosmos3-Nano/

# Then convert:
python -m cosmos_framework.scripts.convert_model_to_dcp \
    -o examples/checkpoints/Cosmos3-Nano \
    --checkpoint-path Cosmos3-Nano
```

---

## Step 4 — Create LoRA Training Config

Create `examples/toml/sft_config/vision_sft_nano_lora_so101.toml`:

```toml
[job]
task = "vfm"
experiment = "vision_sft_nano_lora_so101"
project = "cosmos3"
group = "sft"
name = "so101_lora"
wandb_mode = "disabled"

[model]
max_num_tokens_after_packing = 45056
joint_attn_implementation = "two_way"
precision = "bfloat16"

# LoRA configuration
lora_enabled = true
lora_rank = 16
lora_alpha = 32
lora_target_modules = "q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen"

[model.ema]
enabled = false

[model.parallelism]
data_parallel_shard_degree = -1
data_parallel_replicate_degree = 1
context_parallel_shard_degree = 1

[model.compile]
enabled = false
compile_dynamic = true

[model.activation_checkpointing]
mode = "full"
save_ops_regex = ["fmha"]
preserve_rng_state = true
determinism_check = "default"

[model.tokenizer]
vae_path = "${oc.env:WAN_VAE_PATH}"

[optimizer]
betas = [0.9, 0.95]
eps = 1.0e-6
fused = true
keys_to_select = ["lora_"]
lr = 5.0e-4
weight_decay = 0

[scheduler]
cycle_lengths = [1000]
f_max = [1.0]
f_min = [0.0]
f_start = [0.0]
warm_up_steps = [50]

[trainer]
distributed_parallelism = "fsdp"
grad_accum_iter = 2
logging_iter = 1
max_iter = 500

[trainer.callbacks.grad_clip]
clip_norm = 0.1
force_finite = true

[checkpoint]
keys_to_skip_loading = ["net_ema.", "lora_"]
load_path = "${oc.env:BASE_CHECKPOINT_PATH}"
save_iter = 100
strict_resume = false

[dataloader_train]
max_sequence_length = 45056
```

---

## Step 5 — Create Launch Script

Create `examples/launch_sft_so101_lora.sh`:

```bash
#!/bin/bash
set -euo pipefail

TOML_FILE="examples/toml/sft_config/vision_sft_nano_lora_so101.toml"

# Paths (override with env vars if needed)
: "${DATASET_PATH:=examples/data/so101_sft_dataset}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"
: "${WAN_VAE_PATH:=examples/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
: "${NPROC_PER_NODE:=1}"

export LD_LIBRARY_PATH=""
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DATASET_PATH BASE_CHECKPOINT_PATH WAN_VAE_PATH

echo "============================================"
echo "  Cosmos3-Nano LoRA SFT — SO-101"
echo "============================================"
echo "  Dataset:    $DATASET_PATH"
echo "  Checkpoint: $BASE_CHECKPOINT_PATH"
echo "  VAE:        $WAN_VAE_PATH"
echo "  GPUs:       $NPROC_PER_NODE"
echo "============================================"

IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-outputs}" \
PYTHONPATH=. \
torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_port=50012 \
    -m cosmos_framework.scripts.train \
    --sft-toml="$TOML_FILE" \
    "$@"
```

```bash
chmod +x examples/launch_sft_so101_lora.sh
```

---

## Step 6 — Run Training

### Single GPU (RTX PRO 6000 96GB)

```bash
NPROC_PER_NODE=1 bash examples/launch_sft_so101_lora.sh
```

### Multi-GPU (2x or 4x GPUs)

```bash
NPROC_PER_NODE=2 bash examples/launch_sft_so101_lora.sh
# or
NPROC_PER_NODE=4 bash examples/launch_sft_so101_lora.sh
```

### Monitor Training

```bash
# Watch the log
tail -f outputs/train/cosmos3/sft/so101_lora/logs/*.log

# Or with Weights & Biases (change wandb_mode in TOML)
# wandb_mode = "online"
```

### Training Output Structure

```
outputs/train/cosmos3/sft/so101_lora/
├── config.yaml                    # Resolved config
├── config.pkl                     # Pickled config
├── logs/                          # Training logs
├── checkpoints/
│   ├── latest_checkpoint.txt      # Points to latest iter
│   ├── iter_000000100/            # Checkpoint at step 100
│   ├── iter_000000200/
│   ├── iter_000000300/
│   ├── iter_000000400/
│   └── iter_000000500/            # Final checkpoint
└── wandb/                         # W&B logs (if enabled)
```

### Resume from Checkpoint

```bash
export BASE_CHECKPOINT_PATH="outputs/train/cosmos3/sft/so101_lora/checkpoints/iter_000000300"
NPROC_PER_NODE=1 bash examples/launch_sft_so101_lora.sh
```

---

## Step 7 — Export Trained Model

```bash
# Find latest checkpoint
RUN_DIR=outputs/train/cosmos3/sft/so101_lora
CHECKPOINT_ITER=$(cat $RUN_DIR/checkpoints/latest_checkpoint.txt)
CHECKPOINT_PATH=$RUN_DIR/checkpoints/$CHECKPOINT_ITER

echo "Exporting checkpoint: $CHECKPOINT_PATH"

# Export to HuggingFace safetensors format
python -m cosmos_framework.scripts.export_model \
    --checkpoint-path $CHECKPOINT_PATH \
    --config-file $RUN_DIR/config.yaml \
    -o $RUN_DIR/model
```

---

## Step 8 — Deploy Fine-tuned Model on GB10

Copy the exported model back to your GB10:

```bash
# From training machine:
rsync -avP outputs/train/cosmos3/sft/so101_lora/model/ \
    gb10:/home/dgx-destro/cosmos3/Cosmos3-Nano-SO101/
```

On GB10, use it just like the base model:

```bash
source ~/cosmos3/activate.sh
export TORCH_COMPILE_DISABLE=1
```

### For video generation (diffusers):

```python
pipe = Cosmos3OmniPipeline.from_pretrained(
    "/home/dgx-destro/cosmos3/Cosmos3-Nano-SO101",  # fine-tuned model
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    enable_safety_checker=False,
)
```

### For inverse dynamics (cosmos-framework):

```bash
python -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i ~/cosmos3/inputs/so101_inverse.json \
    -o ~/cosmos3/output/so101_finetuned \
    --checkpoint-path ~/cosmos3/Cosmos3-Nano-SO101 \
    --no-guardrails \
    --seed=0
```

---

## Training Hyperparameters Reference

| Parameter | LoRA (Nano) | Full SFT (Nano) |
|---|---|---|
| **Learning rate** | 5.0e-4 | 2.0e-5 |
| **LoRA rank** | 16 | N/A |
| **LoRA alpha** | 32 | N/A |
| **LoRA targets** | q/k/v/o_proj_moe_gen | N/A |
| **Optimizer** | AdamW (fused) | AdamW (fused) |
| **Betas** | (0.9, 0.95) | (0.9, 0.95) |
| **Weight decay** | 0 | 0 |
| **Grad clip norm** | 0.1 | 0.1 |
| **Warmup steps** | 50 | 50 |
| **Scheduler** | Cosine (LambdaCosine) | Cosine (LambdaCosine) |
| **Max iterations** | 500 | 500 |
| **Grad accum** | 2 | 2 |
| **Checkpoint save** | Every 100 steps | Every 100 steps |
| **Precision** | BF16 | BF16 |
| **EMA** | Disabled | Enabled (rate 0.1) |
| **torch.compile** | Disabled | Enabled |
| **Activation ckpt** | Full | Full |
| **Trainable params** | ~256 MB | ~32 GB |

## Memory Estimates

| Config | 1x RTX PRO 6000 (96GB) | 2x RTX 6000 Ada (48GB) | 8x H100 (80GB) |
|---|---|---|---|
| LoRA Nano (16B) | ~45-55 GB — fits | FSDP shard — fits | fits easily |
| Full SFT Nano (16B) | ~140 GB — no | FSDP shard — tight | fits |
| LoRA Super (64B) | ~90 GB — tight | no | fits |

## Scaling Tips

- **More episodes** = better. 50-100 episodes is a good starting point.
- **Diverse tasks** help generalization. Record picking, placing, sorting.
- **Multiple camera angles** improve robustness. Use both ego and external.
- **Dense captions** matter. Describe the motion, not just the task.
- **500 steps** with 75 episodes is a reasonable starting point. Monitor loss convergence.
- **Lower LR** (1e-4) if training is unstable, **higher** (1e-3) if loss plateaus.

---

## Quick Reference Commands

```bash
# Full pipeline on single RTX PRO 6000:

# 1. Prep data
python convert_lerobot_to_jsonl.py

# 2. Convert checkpoint
python -m cosmos_framework.scripts.convert_model_to_dcp \
    -o examples/checkpoints/Cosmos3-Nano \
    --checkpoint-path Cosmos3-Nano

# 3. Train (single GPU LoRA)
NPROC_PER_NODE=1 bash examples/launch_sft_so101_lora.sh

# 4. Export
python -m cosmos_framework.scripts.export_model \
    --checkpoint-path outputs/train/cosmos3/sft/so101_lora/checkpoints/iter_000000500 \
    --config-file outputs/train/cosmos3/sft/so101_lora/config.yaml \
    -o outputs/train/cosmos3/sft/so101_lora/model

# 5. Deploy to GB10
rsync -avP outputs/train/cosmos3/sft/so101_lora/model/ gb10:~/cosmos3/Cosmos3-Nano-SO101/
```
