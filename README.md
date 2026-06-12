# Cosmos3-Nano

Running NVIDIA Cosmos3-Nano (16B parameters) on NVIDIA Blackwell aarch64 platforms — Project DIGITS (GB10) and Jetson Thor.

https://github.com/user-attachments/assets/f5d6a0a9-4f3f-4103-bbfd-94b0356c2c5e

NVIDIA's recommended setup for Cosmos3 is 8x H100 GPUs. This project runs the full 16B-parameter model on a single desktop device with 128GB unified memory, with text-to-video, image-to-video, and text-to-image generation at up to 720p.

## Capabilities

| Mode | Input | Output |
|---|---|---|
| **Text-to-Video** | Text prompt | MP4 video (up to 720p, configurable length) |
| **Image-to-Video** | Image + optional text | MP4 video |
| **Text-to-Image** | Text prompt | JPG image |
| **Web UI** | Browser interface | All of the above via Gradio |

## Supported Platforms

| Platform | GPU | Memory | CUDA |
|---|---|---|---|
| **GB10** (Project DIGITS) | Blackwell GB10, Compute 12.1 | 128GB unified CPU+GPU | 13.0 |
| **Jetson Thor** | Blackwell Thor SoC | Up to 128GB | 13.x |

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/kabilankb/cosmos3.git
cd cosmos3
```

### 2. Download Model Weights (33GB)

```bash
pip install huggingface_hub hf_transfer

HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/Cosmos3-Nano', local_dir='Cosmos3-Nano')
snapshot_download('nvidia/Cosmos3-Nano', allow_patterns='assets/*', local_dir='Cosmos3-Nano-assets')
"
```

### 3. Clone cosmos-framework

```bash
git clone https://github.com/NVIDIA/cosmos-framework.git
python docker/patches/apply_patches.py cosmos-framework
```

### 4a. Docker (Recommended)

```bash
# Build
docker/build.sh gb10             # or: jetson-thor

# Launch Web UI
docker/run.sh gb10 webui         # http://localhost:7860

# Or interactive shell
docker/run.sh gb10 interactive
```

### 4b. Bare Metal

```bash
# Create venv and install dependencies
python -m venv .venv
source activate.sh
pip install "diffusers @ git+https://github.com/huggingface/diffusers.git" \
    accelerate av huggingface_hub imageio imageio-ffmpeg \
    torch torchvision transformers gradio
pip install -e cosmos-framework/

# Launch Web UI
python webui.py                  # http://localhost:7860

# Or generate interactively
python generate_video.py
python generate_image.py
```

---

## Web UI

Browser-based interface with NVIDIA-themed dark UI. Supports all generation modes with configurable resolution, frames, FPS, inference steps, guidance scale, and seed.

```bash
# Docker
docker/run.sh gb10 webui              # default port 7860
docker/run.sh gb10 webui 8080         # custom port
./webui.sh                             # shortcut

# Docker Compose
cd docker && docker compose up cosmos3-webui

# Bare metal
python webui.py
```

---

## Scripts

| Script | Description |
|---|---|
| `webui.py` | Gradio web interface (T2V, I2V, T2I) |
| `generate_video.py` | Interactive text-to-video and image-to-video |
| `generate_image.py` | Interactive text-to-image |
| `test_t2v.py` | Text-to-video automated test (GB10) |
| `test_i2v.py` | Image-to-video automated test (GB10) |
| `thor_test_t2v.py` | Text-to-video automated test (Jetson Thor) |
| `thor_test_i2v.py` | Image-to-video automated test (Jetson Thor) |

---

## Docker

See [docker/README.md](docker/README.md) for full Docker documentation.

```bash
# Build
docker/build.sh gb10                   # GB10
docker/build.sh jetson-thor            # Jetson Thor
docker/build.sh thor-vllm              # Thor + vLLM

# Run
docker/run.sh <platform> <mode> [port]
#   Modes: interactive | webui | generate | t2v
```

---

## Performance

| Task | Resolution | Time (GB10) |
|---|---|---|
| Text-to-Video (57 frames, 35 steps) | 480p 16:9 | ~3 min |
| Image-to-Video (57 frames, 35 steps) | 480p 16:9 | ~3 min |
| Text-to-Image (35 steps) | 480p 16:9 | ~30 sec |
| Model load | - | ~5 sec |

---

## Supported Resolutions

| Resolution | 16:9 | 4:3 | 1:1 | 9:16 |
|---|---|---|---|---|
| 256p | 456x256 | 336x256 | 256x256 | 256x456 |
| 480p | 848x480 | 640x480 | 480x480 | 480x848 |
| 720p | 1280x720 | 960x720 | 720x720 | 720x1280 |

---

## Project Structure

```
cosmos3/
├── README.md
├── webui.py                        # Gradio Web UI
├── webui.sh                        # Docker Web UI shortcut
├── generate_video.py               # Interactive T2V & I2V generator
├── generate_image.py               # Interactive T2I generator
├── test_t2v.py                     # Text-to-video test (GB10)
├── test_i2v.py                     # Image-to-video test (GB10)
├── thor_test_t2v.py                # Text-to-video test (Jetson Thor)
├── thor_test_i2v.py                # Image-to-video test (Jetson Thor)
├── activate.sh                     # Environment activation
├── docker/
│   ├── Dockerfile.gb10             # GB10 container
│   ├── Dockerfile.jetson-thor      # Jetson Thor container
│   ├── Dockerfile.jetson-thor-vllm # Thor + vLLM container
│   ├── docker-compose.yml          # Compose config
│   ├── build.sh                    # Build helper
│   ├── run.sh                      # Run helper
│   ├── README.md                   # Docker documentation
│   └── patches/
│       ├── apply_patches.py        # Auto-patcher for cosmos-framework
│       └── sdpa_fallback.py        # PyTorch SDPA attention backend
├── COSMOS3_GB10_GUIDE.md           # GB10 detailed setup guide
├── COSMOS3_JETSON_THOR_GUIDE.md    # Jetson Thor setup guide
├── BLOG_COSMOS3_GB10.md            # Blog post
├── Cosmos3-Nano/                   # Model weights (33GB, not in repo)
├── Cosmos3-Nano-assets/            # Example prompts (not in repo)
└── cosmos-framework/               # NVIDIA framework (cloned separately)
```

---

## Platform Guides

- [GB10 Setup Guide](COSMOS3_GB10_GUIDE.md) — Complete guide for NVIDIA Project DIGITS
- [Jetson Thor Setup Guide](COSMOS3_JETSON_THOR_GUIDE.md) — Complete guide for Jetson Thor
- [Blog Post](BLOG_COSMOS3_GB10.md) — Full story of running Cosmos3 on GB10

## Software Stack

| Component | Version |
|---|---|
| Python | 3.13 |
| PyTorch | 2.12 + CUDA 13.0 |
| Diffusers | 0.39+ (from git) |
| cosmos-framework | 1.2.2 (patched) |
| Gradio | 6.x |
| Model | Cosmos3-Nano (16B params, ~33GB) |

## Links

- [Cosmos3-Nano on HuggingFace](https://huggingface.co/nvidia/Cosmos3-Nano)
- [Cosmos3 Model Collection](https://huggingface.co/collections/nvidia/cosmos3)
- [cosmos-framework](https://github.com/NVIDIA/cosmos-framework)
- [NVIDIA Cosmos](https://github.com/NVIDIA/cosmos)
- [Technical Report](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf)

## License

This project uses NVIDIA Cosmos3-Nano under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
