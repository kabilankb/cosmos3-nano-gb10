# Cosmos3-Nano Docker — GB10 & Jetson Thor

Containerized Cosmos3-Nano (16B) for video generation and robot action trajectory extraction on NVIDIA Blackwell aarch64 platforms.

## Supported Platforms

| Platform | GPU | Memory | Base Image | CUDA |
|---|---|---|---|---|
| **GB10** (Project DIGITS) | Blackwell GB10 | 128GB unified | `nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04` | 13.0 |
| **Jetson Thor** | Blackwell Thor SoC | Up to 128GB | `nvcr.io/nvidia/l4t-pytorch:r37.1.0-pth2.7-py3` | 13.x |

---

## 1. Prerequisites

### Install Docker + NVIDIA Container Toolkit

```bash
# Docker (if not installed)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Download Model Weights (33GB, one-time)

```bash
pip install huggingface_hub hf_transfer

HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/Cosmos3-Nano', local_dir='$HOME/cosmos3/Cosmos3-Nano')
"
```

### Download Example Assets

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/Cosmos3-Nano', allow_patterns='assets/*', local_dir='$HOME/cosmos3/Cosmos3-Nano-assets')
"
```

---

## 2. Build

```bash
cd ~/cosmos3

# GB10
docker/build.sh gb10

# Jetson Thor
docker/build.sh jetson-thor

# Both platforms
docker/build.sh both
```

### Or build manually

```bash
cd ~/cosmos3

# GB10
docker build -f docker/Dockerfile.gb10 -t cosmos3-nano:gb10 .

# Jetson Thor
docker build -f docker/Dockerfile.jetson-thor -t cosmos3-nano:jetson-thor .
```

---

## 3. Run

### Using the run script

```bash
docker/run.sh <platform> <mode>
```

| Platform | Mode | Description |
|---|---|---|
| `gb10` | `interactive` | Interactive bash shell |
| `gb10` | `generate` | Image-to-video generator (interactive prompts) |
| `gb10` | `trajectory` | Inverse dynamics trajectory extraction (interactive) |
| `gb10` | `t2v` | Text-to-video test (automated) |
| `jetson-thor` | `interactive` | Interactive bash shell |
| `jetson-thor` | `generate` | Image-to-video generator |
| `jetson-thor` | `trajectory` | Trajectory extraction |
| `jetson-thor` | `t2v` | Text-to-video test |

### Examples

```bash
# Interactive shell on GB10
docker/run.sh gb10 interactive

# Generate video from image on GB10
docker/run.sh gb10 generate

# Extract robot trajectory on GB10
docker/run.sh gb10 trajectory

# Run text-to-video test on GB10
docker/run.sh gb10 t2v

# Same commands on Jetson Thor
docker/run.sh jetson-thor interactive
docker/run.sh jetson-thor generate
docker/run.sh jetson-thor trajectory
```

---

## 4. Raw Docker Commands

If you prefer running docker directly without the helper scripts:

### Interactive Shell

```bash
docker run --rm -it \
    --runtime=nvidia --gpus all --ipc=host \
    -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    -e TORCH_COMPILE_DISABLE=1 \
    -e LD_LIBRARY_PATH="" \
    -v ~/cosmos3/Cosmos3-Nano:/workspace/cosmos3/Cosmos3-Nano:ro \
    -v ~/cosmos3/Cosmos3-Nano-assets:/workspace/cosmos3/Cosmos3-Nano-assets:ro \
    -v ~/cosmos3/output:/workspace/cosmos3/output \
    -v ~/cosmos3/inputs:/workspace/cosmos3/inputs \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    cosmos3-nano:gb10
```

### Image-to-Video (interactive)

```bash
docker run --rm -it \
    --runtime=nvidia --gpus all --ipc=host \
    -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    -e TORCH_COMPILE_DISABLE=1 \
    -e LD_LIBRARY_PATH="" \
    -v ~/cosmos3/Cosmos3-Nano:/workspace/cosmos3/Cosmos3-Nano:ro \
    -v ~/cosmos3/Cosmos3-Nano-assets:/workspace/cosmos3/Cosmos3-Nano-assets:ro \
    -v ~/cosmos3/output:/workspace/cosmos3/output \
    -v ~/cosmos3/inputs:/workspace/cosmos3/inputs \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    cosmos3-nano:gb10 \
    bash -c "source .venv/bin/activate && python generate_video.py"
```

### Trajectory Extraction (interactive)

```bash
docker run --rm -it \
    --runtime=nvidia --gpus all --ipc=host \
    -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    -e TORCH_COMPILE_DISABLE=1 \
    -e LD_LIBRARY_PATH="" \
    -v ~/cosmos3/Cosmos3-Nano:/workspace/cosmos3/Cosmos3-Nano:ro \
    -v ~/cosmos3/Cosmos3-Nano-assets:/workspace/cosmos3/Cosmos3-Nano-assets:ro \
    -v ~/cosmos3/output:/workspace/cosmos3/output \
    -v ~/cosmos3/inputs:/workspace/cosmos3/inputs \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    cosmos3-nano:gb10 \
    bash -c "source .venv/bin/activate && python extract_trajectory.py"
```

### Text-to-Video Test (automated)

```bash
docker run --rm \
    --runtime=nvidia --gpus all --ipc=host \
    -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    -e TORCH_COMPILE_DISABLE=1 \
    -e LD_LIBRARY_PATH="" \
    -v ~/cosmos3/Cosmos3-Nano:/workspace/cosmos3/Cosmos3-Nano:ro \
    -v ~/cosmos3/Cosmos3-Nano-assets:/workspace/cosmos3/Cosmos3-Nano-assets:ro \
    -v ~/cosmos3/output:/workspace/cosmos3/output \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    cosmos3-nano:gb10 \
    bash -c "source .venv/bin/activate && python test_t2v.py"
```

### Inverse Dynamics with JSON Config (non-interactive)

```bash
# Create input JSON on host first
cat > ~/cosmos3/inputs/my_run.json << 'EOF'
{
  "action_chunk_size": 16,
  "domain_name": "lerobot-so101",
  "fps": 10,
  "image_size": 480,
  "view_point": "ego_view",
  "model_mode": "inverse_dynamics",
  "name": "my_run",
  "prompt": "A SO-101 robot arm picking a part",
  "seed": 0,
  "vision_path": "/workspace/cosmos3/output/my_video.mp4"
}
EOF

# Run inference
docker run --rm \
    --runtime=nvidia --gpus all --ipc=host \
    -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    -e TORCH_COMPILE_DISABLE=1 \
    -e LD_LIBRARY_PATH="" \
    -v ~/cosmos3/Cosmos3-Nano:/workspace/cosmos3/Cosmos3-Nano:ro \
    -v ~/cosmos3/output:/workspace/cosmos3/output \
    -v ~/cosmos3/inputs:/workspace/cosmos3/inputs \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    cosmos3-nano:gb10 \
    bash -c "source .venv/bin/activate && \
    cd cosmos-framework && \
    python -m cosmos_framework.scripts.inference \
        --parallelism-preset=latency \
        -i /workspace/cosmos3/inputs/my_run.json \
        -o /workspace/cosmos3/output/my_run_output \
        --checkpoint-path /workspace/cosmos3/Cosmos3-Nano \
        --no-guardrails --seed=0"
```

---

## 5. Docker Compose

```bash
cd ~/cosmos3/docker

# GB10 interactive
docker compose run cosmos3-gb10

# Jetson Thor interactive
docker compose run cosmos3-jetson-thor

# Tear down
docker compose down
```

---

## 6. Inside the Container

Once inside the container, all scripts are at `/workspace/cosmos3/`:

```bash
# Activate environment
source .venv/bin/activate

# Generate video from image (interactive)
python generate_video.py

# Extract trajectory from video (interactive)
python extract_trajectory.py

# Text-to-video test
python test_t2v.py

# Image-to-video test
python test_i2v.py

# Direct cosmos-framework inference
cd cosmos-framework
python -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i /workspace/cosmos3/inputs/input.json \
    -o /workspace/cosmos3/output/result \
    --checkpoint-path /workspace/cosmos3/Cosmos3-Nano \
    --no-guardrails --seed=0
```

---

## 7. Volume Mounts

| Container Path | Host Path | Mode | Purpose |
|---|---|---|---|
| `/workspace/cosmos3/Cosmos3-Nano` | `~/cosmos3/Cosmos3-Nano` | ro | Model weights (33GB) |
| `/workspace/cosmos3/Cosmos3-Nano-assets` | `~/cosmos3/Cosmos3-Nano-assets` | ro | Example prompts & inputs |
| `/workspace/cosmos3/output` | `~/cosmos3/output` | rw | Generated videos & trajectories |
| `/workspace/cosmos3/inputs` | `~/cosmos3/inputs` | rw | Input JSON configs |
| `/root/.cache/huggingface` | `~/.cache/huggingface` | rw | HF cache (Qwen3 tokenizer, VAE) |

Output files are written to `~/cosmos3/output/` on the host and persist after the container exits.

---

## 8. Environment Variables

| Variable | Value | Purpose |
|---|---|---|
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Prevents CUDA OOM fragmentation |
| `TORCH_COMPILE_DISABLE` | `1` | Disables torch.compile (fails on GB10/Thor attention) |
| `LD_LIBRARY_PATH` | `""` | Cleared to avoid library conflicts |
| `HF_HUB_ENABLE_HF_TRANSFER` | `1` | Fast Rust-based HF downloads |

---

## 9. Patches Baked In

Both containers include 4 patches for Blackwell aarch64 compatibility:

| Patch | File | Issue | Fix |
|---|---|---|---|
| 1 | `inference/args.py` | `nvmlDeviceGetMemoryInfo` unsupported | Fallback to `torch.cuda` |
| 2 | `inference/vision.py` | `torchvision.io.read_video` removed v0.27 | PyAV reader |
| 3 | `model/attention/sdpa_fallback.py` | No flash-attn/natten | PyTorch SDPA with GQA |
| 4 | `data/vfm/action/domain_utils.py` | SO-101 not registered | `lerobot-so101` 6D |

To apply these patches on bare metal (no Docker):

```bash
python ~/cosmos3/docker/patches/apply_patches.py ~/cosmos3/cosmos-framework
```

---

## 10. Registered Robot Embodiments

| Domain Name | Dim | Robot |
|---|---|---|
| `lerobot-so101` | 6D | LeRobot SO-101 (5 joints + gripper) |
| `robomind-ur` | 10D | UR robot |
| `robomind-franka` | 10D | Franka Panda + RobotiQ |
| `robomind-franka-dual` | 20D | Dual Franka Panda |
| `bridge_orig_lerobot` | 10D | WidowX 250 |
| `droid_lerobot` | 10D | DROID |
| `agibotworld` | 29D | Agibot |
| `umi` | 10D | UMI |
| `fractal` | 10D | Google robot |
| `av` | 9D | Autonomous vehicle |
| `camera_pose` | 9D | Camera motion |

---

## 11. Performance

| Task | Steps | Speed | Total |
|---|---|---|---|
| Text-to-video (480p, 57 frames) | 35 | ~5.3 s/step | ~3 min |
| Image-to-video (480p, 57 frames) | 35 | ~5.3 s/step | ~3 min |
| Inverse dynamics (16 frames) | 30 | ~1.4 it/s | ~20 sec |
| Model load (cached) | - | - | ~5 sec |

---

## 12. Troubleshooting

### Container can't see GPU

```bash
# Verify nvidia-container-toolkit
nvidia-container-cli info

# Test GPU access
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi
```

### Out of memory

Reduce resolution and frames inside the container:

```python
# In generate_video.py or test scripts, use:
height=256, width=456, num_frames=25, num_inference_steps=20
```

### Model not found

Ensure model weights are downloaded and the volume mount is correct:

```bash
ls ~/cosmos3/Cosmos3-Nano/*.safetensors | wc -l
# Should show 10 files
```

### Jetson Thor base image not available

The L4T image tag `r37.1.0-pth2.7-py3` is a placeholder for JetPack 7.x.
Check available images at: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch

For Jetson Orin (JetPack 6.x), edit the Dockerfile:

```dockerfile
FROM nvcr.io/nvidia/l4t-pytorch:r36.4.0-pth2.5-py3 AS base
```

---

## 13. File Reference

```
~/cosmos3/
├── docker/
│   ├── Dockerfile.gb10                 # GB10 container
│   ├── Dockerfile.jetson-thor          # Jetson Thor container
│   ├── docker-compose.yml              # Compose for both
│   ├── build.sh                        # Build helper
│   ├── run.sh                          # Run helper
│   ├── patches/
│   │   ├── apply_patches.py            # Auto-patcher
│   │   └── sdpa_fallback.py            # SDPA backend
│   └── README.md                       # This file
├── Cosmos3-Nano/                       # Model weights (33GB)
├── Cosmos3-Nano-assets/assets/         # Example prompts
├── cosmos-framework/                   # NVIDIA framework (patched)
├── output/                             # Generated output
├── inputs/                             # Input JSON configs
├── generate_video.py                   # Interactive i2v
├── extract_trajectory.py               # Interactive inverse dynamics
├── test_t2v.py                         # Text-to-video test
├── test_i2v.py                         # Image-to-video test
├── activate.sh                         # Env activation (bare metal)
├── COSMOS3_GB10_GUIDE.md               # Full setup guide
└── COSMOS3_FINETUNE_GUIDE.md           # Fine-tuning guide
```
