# cosmos3-nano-gb10

Running NVIDIA Cosmos3-Nano (16B parameters) on a single NVIDIA Project DIGITS (GB10) desktop AI supercomputer.

https://github.com/user-attachments/assets/f5d6a0a9-4f3f-4103-bbfd-94b0356c2c5e

## Overview

NVIDIA's recommended setup for Cosmos3 is 8x H100 GPUs. This project demonstrates running the 16B-parameter Cosmos3-Nano on a single GB10 — a $3,000 desktop device — with full text-to-video, image-to-video, and text-to-image generation at 480p/24fps.

## Hardware

| Spec | Value |
|---|---|
| GPU | NVIDIA GB10 (Blackwell, Compute 12.1) |
| Memory | 128GB unified CPU+GPU |
| CUDA | 13.0 |
| Architecture | aarch64 (ARM) |

The 128GB unified memory is the key enabler. Cosmos3-Nano's weights are ~33GB in BF16, fitting comfortably with ~95GB remaining for activations, KV cache, and the VAE decoder.

## Capabilities

- **Text-to-Video** — Generate video from text prompts
- **Image-to-Video** — Animate a static image into video
- **Text-to-Image** — Generate images from text descriptions
- **Action Generation** — Extract and generate robotic actions from video (with vLLM-Omni)

## Software Stack

| Component | Version |
|---|---|
| Python | 3.13 |
| PyTorch | 2.12 + CUDA 13.0 |
| Diffusers | 0.39 (from git) |
| cosmos-framework | 1.2.2 (patched for GB10) |
| Model | Cosmos3-Nano (16B params, ~33GB) |

## Project Structure

```
cosmos3-nano-gb10/
├── README.md
├── BLOG_COSMOS3_GB10.md              # Blog post — full story of running Cosmos3 on GB10
├── COSMOS3_GB10_GUIDE.md             # Complete setup guide with patches and usage
├── activate.sh                       # Quick environment activation script
├── generate_video.py                 # Interactive text-to-video & image-to-video generator
├── generate_image.py                 # Interactive text-to-image generator
├── test_t2v.py                       # Text-to-video example
├── test_i2v.py                       # Image-to-video example
├── action/
│   ├── docs/
│   │   ├── COSMOS3_FINETUNE_GUIDE.md     # Fine-tuning guide for Cosmos3
│   │   └── COSMOS3_POLICY_DROID_GUIDE.md # Policy learning with Droid dataset
│   ├── inputs/
│   │   ├── inverse_dynamics_test.json    # Test config for action extraction
│   │   ├── pen_lerobot.json              # LeRobot pen manipulation config
│   │   ├── robot_pick_inverse.json       # Robot pick-and-place config
│   │   └── so101_inverse.json            # SO-101 robot arm config
│   └── scripts/
│       ├── extract_actions.py            # Action extraction from video (vLLM-Omni)
│       ├── extract_trajectory.py         # Trajectory extraction pipeline
│       └── start_server.sh              # vLLM-Omni Docker server launcher
└── docker/
    ├── README.md                         # Docker setup instructions
    ├── Dockerfile.gb10                   # Dockerfile for NVIDIA Project DIGITS
    ├── Dockerfile.jetson-thor            # Dockerfile for Jetson Thor
    ├── docker-compose.yml                # Compose config for multi-container setup
    ├── build.sh                          # Docker build script
    ├── run.sh                            # Docker run script
    └── patches/
        ├── apply_patches.py              # Auto-apply GB10 patches to cosmos-framework
        └── sdpa_fallback.py              # PyTorch native SDPA attention backend
```

## GB10 Patches

The NVIDIA cosmos-framework required four patches for GB10 compatibility:

1. **Memory detection** — `nvmlDeviceGetMemoryInfo` not supported on GB10; fallback to `torch.cuda.get_device_properties()`
2. **Video I/O** — `torchvision.io.read_video` removed in v0.27; replaced with PyAV
3. **Attention backend** — No flash-attn/natten on GB10; added PyTorch native SDPA with GQA head expansion
4. **Scheduler config** — Diffusers `UniPCMultistepScheduler` compatibility fix

## Performance

| Task | Resolution | Time |
|---|---|---|
| Text-to-Video (57 frames) | 480p 16:9 | ~8 min |
| Image-to-Video (57 frames) | 480p 16:9 | ~8 min |
| Text-to-Image | 480p 16:9 | ~30 sec |

## License

This project uses NVIDIA Cosmos3-Nano under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
