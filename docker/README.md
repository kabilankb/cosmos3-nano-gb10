# Cosmos3-Nano Docker — GB10 & Jetson Thor

Containerized Cosmos3-Nano (16B) for video/image generation on NVIDIA Blackwell aarch64 platforms.

## Supported Platforms

| Platform | GPU | Memory | Base Image | CUDA |
|---|---|---|---|---|
| **GB10** (Project DIGITS) | Blackwell GB10 | 128GB unified | `nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04` | 13.0 |
| **Jetson Thor** | Blackwell Thor SoC | Up to 128GB | `nvcr.io/nvidia/l4t-pytorch:r37.1.0-pth2.7-py3` | 13.x |
| **Thor + vLLM** | Blackwell Thor SoC | Up to 128GB | `thor_vllm_container:25.08-py3-base` | 13.x |

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
snapshot_download('nvidia/Cosmos3-Nano', local_dir='Cosmos3-Nano')
"
```

### Download Example Assets

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/Cosmos3-Nano', allow_patterns='assets/*', local_dir='Cosmos3-Nano-assets')
"
```

---

## 2. Build

```bash
docker/build.sh gb10           # GB10
docker/build.sh jetson-thor    # Jetson Thor
docker/build.sh thor-vllm      # Thor + vLLM
docker/build.sh both           # GB10 + Jetson Thor
```

---

## 3. Run

```bash
docker/run.sh <platform> <mode> [port]
```

| Platform | Mode | Description |
|---|---|---|
| `gb10` | `interactive` | Interactive bash shell |
| `gb10` | `webui` | Web UI (Gradio) |
| `gb10` | `generate` | Video generator (interactive prompts) |
| `gb10` | `t2v` | Text-to-video test (automated) |
| `jetson-thor` | `interactive` | Interactive bash shell |
| `jetson-thor` | `webui` | Web UI (Gradio) |
| `jetson-thor` | `generate` | Video generator |
| `jetson-thor` | `t2v` | Text-to-video test |
| `thor-vllm` | `interactive` | Interactive bash shell |
| `thor-vllm` | `webui` | Web UI (Gradio) |

### Examples

```bash
# Web UI on GB10 (default port 7860)
docker/run.sh gb10 webui

# Web UI on custom port
docker/run.sh gb10 webui 8080

# Web UI shortcut
./webui.sh
./webui.sh 8080

# Interactive shell
docker/run.sh gb10 interactive

# Generate video (interactive prompts)
docker/run.sh gb10 generate

# Automated text-to-video test
docker/run.sh gb10 t2v

# Same commands work with jetson-thor
docker/run.sh jetson-thor webui
docker/run.sh jetson-thor interactive
```

---

## 4. Docker Compose

```bash
cd docker

# Web UI
docker compose up cosmos3-webui

# Interactive shell (GB10)
docker compose run cosmos3-gb10

# Interactive shell (Jetson Thor)
docker compose run cosmos3-jetson-thor

# Tear down
docker compose down
```

---

## 5. Inside the Container

All scripts are at `/workspace/cosmos3/`:

```bash
# Activate environment
source .venv/bin/activate

# Web UI
python webui.py

# Generate video (interactive)
python generate_video.py

# Generate image (interactive)
python generate_image.py

# Text-to-video test
python test_t2v.py

# Image-to-video test
python test_i2v.py
```

---

## 6. Volume Mounts

| Container Path | Host Path | Mode | Purpose |
|---|---|---|---|
| `/workspace/cosmos3/Cosmos3-Nano` | `./Cosmos3-Nano` | ro | Model weights (33GB) |
| `/workspace/cosmos3/Cosmos3-Nano-assets` | `./Cosmos3-Nano-assets` | ro | Example prompts & inputs |
| `/workspace/cosmos3/output` | `./output` | rw | Generated videos & images |
| `/workspace/cosmos3/inputs` | `./inputs` | rw | Input configs |
| `/root/.cache/huggingface` | `~/.cache/huggingface` | rw | HF cache (tokenizer, VAE) |

Output files persist on the host after the container exits.

---

## 7. Environment Variables

| Variable | Value | Purpose |
|---|---|---|
| `COSMOS3_DIR` | `/workspace/cosmos3` | Base path for model/assets/output |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Prevents CUDA OOM fragmentation |
| `TORCH_COMPILE_DISABLE` | `1` | Disables torch.compile (fails on Blackwell attention) |
| `LD_LIBRARY_PATH` | `""` | Cleared to avoid library conflicts |
| `HF_HUB_ENABLE_HF_TRANSFER` | `1` | Fast Rust-based HF downloads |

---

## 8. Patches Baked In

Both containers include patches for Blackwell aarch64 compatibility:

| Patch | File | Fix |
|---|---|---|
| Memory detection | `inference/args.py` | Fallback to `torch.cuda` (nvml unsupported) |
| Video I/O | `inference/vision.py` | PyAV reader (torchvision.io removed in v0.27) |
| Attention backend | `model/attention/sdpa_fallback.py` | PyTorch SDPA with GQA expansion |

To apply on bare metal (no Docker):

```bash
python docker/patches/apply_patches.py cosmos-framework
```

---

## 9. Performance

| Task | Resolution | Time (GB10) |
|---|---|---|
| Text-to-Video (57 frames, 35 steps) | 480p 16:9 | ~3 min |
| Image-to-Video (57 frames, 35 steps) | 480p 16:9 | ~3 min |
| Text-to-Image (35 steps) | 480p 16:9 | ~30 sec |
| Model load | - | ~5 sec |

### If you hit OOM

1. Reduce resolution: `height=256, width=456`
2. Reduce frames: `num_frames=25`
3. Reduce steps: `num_inference_steps=20`

---

## 10. Troubleshooting

### Container can't see GPU

```bash
nvidia-container-cli info
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi
```

### Model not found

Ensure weights are downloaded and volume mount is correct:

```bash
ls Cosmos3-Nano/*.safetensors | wc -l
# Should show 10+ files
```

### Port already in use

```bash
# Find what's using the port
ss -tlnp | grep 7860

# Use a different port
docker/run.sh gb10 webui 7861
```

### Jetson Thor base image not available

The L4T image tag `r37.1.0-pth2.7-py3` is for JetPack 7.x. Check available images at:
https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch

For Jetson Orin (JetPack 6.x), edit the Dockerfile base image:

```dockerfile
FROM nvcr.io/nvidia/l4t-pytorch:r36.4.0-pth2.5-py3 AS base
```
