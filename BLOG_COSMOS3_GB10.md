# Running NVIDIA Cosmos3-Nano on Project DIGITS (GB10): Video Generation from a 16B World Model on Your Desk

NVIDIA's recommended setup for Cosmos3 is 8x H100 GPUs — a $200K+ cluster. I got the 16B-parameter Cosmos3-Nano running on a single NVIDIA Project DIGITS, a $3,000 desktop AI supercomputer. Text-to-video, image-to-video, 480p at 24fps — all on a device that sits next to my monitor.

It wasn't plug and play. The framework needed four patches to work on GB10's Blackwell architecture. Here's the full story.

---

## What is Cosmos3?

Cosmos3 is NVIDIA's omnimodal world foundation model, released May 2026. It's not a chatbot — it's a model that understands and simulates the physical world.

The model family has two tiers:

| Model | Parameters | Hardware Required |
|---|---|---|
| Cosmos3-Super | 64B | 8x H100/H200 GPUs |
| Cosmos3-Nano | 16B | Single GPU (this article) |

Both use a **Mixture-of-Transformers (MoT)** architecture — an autoregressive transformer for text combined with a diffusion transformer for continuous modalities (images, video, audio). The model generates video by iteratively denoising latent representations, guided by richly structured text prompts.

Cosmos3-Nano punches above its weight. Same architecture as Super, same multimodal capabilities — text-to-video, image-to-video, text-to-image — just at a smaller scale.

---

## Why Project DIGITS?

Project DIGITS (GB10) is NVIDIA's desktop AI supercomputer:

| Spec | Value |
|---|---|
| GPU | NVIDIA GB10 (Blackwell, Compute 12.1) |
| Memory | 128GB unified CPU+GPU |
| CUDA | 13.0 |
| Architecture | aarch64 (ARM) |
| Price | ~$3,000 |

The key spec is **128GB unified memory**. Cosmos3-Nano's weights are ~33GB in BF16. On a discrete GPU, you'd need a card with at least 40-48GB of VRAM. With GB10's unified memory architecture, the 33GB model fits comfortably with ~95GB left for activations, KV cache, and the VAE decoder.

No PCIe transfer overhead either — CPU and GPU share the same memory pool.

The 64B Cosmos3-Super needs ~128GB just for weights. That fills the entire memory with no room for inference. Nano is the right fit.

---

## Setting Up the Stack

The environment is Python 3.13, PyTorch 2.12 with CUDA 13.0, and HuggingFace Diffusers installed from the latest git (the Cosmos3 pipeline hasn't hit a stable release yet).

```bash
# Create workspace
mkdir -p ~/cosmos3 && cd ~/cosmos3
uv venv --python 3.13 --seed .venv
source .venv/bin/activate

# Install with CUDA 13.0 backend
uv pip install --torch-backend=cu130 \
  "diffusers @ git+https://github.com/huggingface/diffusers.git" \
  accelerate av torch torchvision transformers
```

Downloading the model is ~33GB. With `hf_transfer` (Rust-based downloader), it took about 35 minutes on my connection. The download pulls 8 safetensor shards in parallel — each around 4.5GB — and you can watch them stream in:

```python
from huggingface_hub import snapshot_download
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
snapshot_download('nvidia/Cosmos3-Nano', local_dir='Cosmos3-Nano')
```

Once downloaded, a quick sanity check:

```python
import torch
print(torch.__version__)          # 2.12.0+cu130
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_name(0))  # NVIDIA GB10
```

---

## First Generation

The Diffusers pipeline is the simplest path to video generation. Load the model, set up the scheduler, and generate:

```python
import json, torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

pipe = Cosmos3OmniPipeline.from_pretrained(
    "Cosmos3-Nano",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    enable_safety_checker=False,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    num_frames=57,          # 2.4 seconds at 24fps
    height=480,             # 480p — sweet spot for GB10
    width=848,              # 16:9 aspect ratio
    num_inference_steps=35,
    guidance_scale=6.0,
    generator=torch.Generator(device="cuda").manual_seed(42),
)

export_to_video(result.video, "output.mp4", fps=24)
```

**Result:** The model loaded all 7 transformer shards in about 3 seconds (unified memory makes loading fast — no PCIe transfer), then generated a 480p video at ~5.3 seconds per denoising step. Total: **about 3 minutes for a 2.4-second clip**.

Not real-time. But usable for batch generation, prototyping scenes, and creating synthetic training data.

---

## Prompt Upsampling: The Secret to Quality

Cosmos3 doesn't work well with simple text prompts like "a robot in a kitchen." It expects richly structured JSON prompts with detailed scene descriptions, camera motion, lighting, subject appearance, and temporal captions.

The framework includes a **prompt upsampler** that uses an LLM (Claude, GPT-4, etc.) to expand a simple sentence into this structured format:

```bash
echo "A robot arm cleaning a plate in the kitchen" > prompt.txt

python -m cosmos_framework.inference.prompt_upsampling \
    --input prompt.txt \
    --output upsampled/ \
    --mode text2video \
    --endpoint-url "https://api.anthropic.com/v1/" \
    --model "claude-sonnet-4-6" \
    --api-token "$ANTHROPIC_API_KEY" \
    --resolution 480 \
    --aspect-ratio "16,9"
```

A simple 10-word prompt becomes a ~2000-word JSON with fields like:

```json
{
  "subjects": [{
    "description": "A modern industrial robotic arm with silver and dark gray metallic body...",
    "action": "Wiping a dirty plate with circular sweeping motions",
    "state_changes": "The arm moves fluidly from one side to the other..."
  }],
  "lighting": {
    "conditions": "Warm overhead kitchen lighting...",
    "shadows": "Soft shadows under the robotic arm base..."
  },
  "cinematography": {
    "camera_motion": "Slow steady dolly from left to right...",
    "depth_of_field": "Shallow focus on the robot end-effector..."
  },
  "temporal_caption": "At 0.0 seconds the arm extends downward..."
}
```

The difference in output quality between raw text and upsampled JSON is dramatic. The upsampling costs pennies (a single LLM API call) and the structured prompt feeds directly into the generation pipeline.

---

## Image-to-Video: Animate Your Photos

Feed Cosmos3 an image plus a description of what should happen, and it generates a video that starts from that image:

```python
from diffusers.utils import load_image

image = load_image("robot_scene.jpg")

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    image=image,
    num_frames=73,      # 3 seconds at 24fps
    height=480,
    width=848,
    num_inference_steps=35,
    guidance_scale=6.0,
)
```

Same ~3 minute generation time. The first frame matches your input image, and the model generates plausible motion from there.

I built an interactive script that prompts for the image path and description in the terminal — no need to write code each time:

```bash
python generate_video.py

# Prompts:
#   Image path: /path/to/photo.jpg
#   Prompt: The robot arm reaches down and picks up the object
#   Duration [3]: 
#   Resolution [480]: 
#   Output [auto]: 
```

---

## The Four Patches: What Broke and Why

The cosmos-framework was built and tested on H100/A100 data center GPUs. Project DIGITS, despite being Blackwell, is architecturally different in ways that matter. Here's what broke:

### Patch 1 — NVML Memory Detection

GB10's unified memory architecture doesn't expose GPU memory through NVML the way discrete GPUs do. The framework calls `nvmlDeviceGetMemoryInfo()` during initialization and crashes with "Not Supported."

**Fix:** Fall back to PyTorch's own device property query:

```python
try:
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.total
except pynvml.NVMLError:
    return torch.cuda.get_device_properties(0).total_memory
```

### Patch 2 — Video Reading API

The framework uses `torchvision.io.read_video()` to load input videos for image-to-video generation. This function was removed in torchvision 0.27, which ships with PyTorch 2.12 (the version we need for CUDA 13.0).

**Fix:** Replace with PyAV, which is already a dependency:

```python
import av
container = av.open(str(path))
frames = [torch.from_numpy(f.to_ndarray(format="rgb24")) 
          for f in container.decode(video=0)]
```

### Patch 3 — Attention Backend (The Critical One)

This was the wall that took the most work. Cosmos3's diffusion transformer uses two attention backends: **Flash Attention 2/3** and **NATTEN** (Neighborhood Attention). Neither has ARM/aarch64 builds for Blackwell.

When the model hits its first denoising step:

```
ValueError: Could not find a compatible Attention backend 
for this use case / device.
```

**Fix:** I wrote a PyTorch native SDPA (Scaled Dot-Product Attention) fallback. The tricky part was handling **grouped-query attention** (GQA) — the model uses 32 query heads but only 8 key/value heads. Standard SDPA needs matching head counts, so the fallback repeats KV heads to match:

```python
def sdpa_attention(query, key, value, is_causal=False, **kwargs):
    h_q, h_kv = query.shape[2], key.shape[2]
    if h_q != h_kv:
        repeat_factor = h_q // h_kv
        key = key.repeat_interleave(repeat_factor, dim=2)
        value = value.repeat_interleave(repeat_factor, dim=2)
    
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    return out.transpose(1, 2)
```

This was registered as a new `"sdpa"` backend in the framework's attention dispatcher, with an always-true compatibility check so it serves as a universal fallback.

### Patch 4 — torch.compile Disabled

Even with the SDPA backend working in eager mode, `torch.compile` (dynamo) fails during JIT compilation of the attention path. The compiled graph can't handle the attention dispatch on GB10.

**Fix:** `export TORCH_COMPILE_DISABLE=1`. Eager mode is slightly slower but fully functional.

### Applying the Patches

All four patches are captured in a single auto-patcher script:

```bash
python docker/patches/apply_patches.py /path/to/cosmos-framework
```

Run it once against a fresh cosmos-framework clone and everything works.

---

## Supported Resolutions and Settings

Cosmos3-Nano supports multiple resolutions and aspect ratios:

| Resolution | 16:9 | 4:3 | 1:1 | 9:16 |
|---|---|---|---|---|
| 256p | 456x256 | 336x256 | 256x256 | 256x456 |
| 480p | 848x480 | 640x480 | 480x480 | 480x848 |
| 720p | 1280x720 | 960x720 | 720x720 | 720x1280 |

On GB10, **480p is the sweet spot**. 720p is possible but tight on memory — reduce inference steps to 25 and frame count to avoid OOM. 256p is fast and useful for quick previews.

If you hit out-of-memory errors:

1. Drop to 256p (`height=256, width=456`)
2. Reduce frames (`num_frames=25`)
3. Reduce steps (`num_inference_steps=20`)
4. Set `export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`

---

## Performance Numbers

| Metric | Value |
|---|---|
| Model loading | ~3 seconds |
| Text-to-video (480p, 57 frames, 35 steps) | ~3 minutes |
| Image-to-video (480p, 57 frames, 35 steps) | ~3 minutes |
| Text-to-video (256p, 25 frames, 20 steps) | ~45 seconds |
| Step speed at 480p | ~5.3 seconds/step |
| Step speed at 256p | ~2.2 seconds/step |

For context, the same model on 8x H200 generates a 720p video in ~55 seconds with 50 steps. GB10 is roughly 10-15x slower — the price of running on a single desktop GPU vs. an 8-GPU cluster.

But the economics are different. An 8x H200 node costs $25-40/hour in the cloud. GB10 sits on your desk and costs nothing after the initial $3,000 investment. For prototyping, batch overnight generation, or creating small datasets of synthetic video, the math works.

---

## Docker: Portable Setup

I packaged everything into Docker containers that work on both GB10 and Jetson Thor (both are Blackwell aarch64):

```bash
# Build
docker build -f docker/Dockerfile.gb10 -t cosmos3-nano:gb10 .

# Run interactive
docker run --rm -it --runtime=nvidia --gpus all --ipc=host \
  -v ~/cosmos3/Cosmos3-Nano:/workspace/cosmos3/Cosmos3-Nano:ro \
  -v ~/cosmos3/output:/workspace/cosmos3/output \
  cosmos3-nano:gb10

# Run video generation directly
docker run --rm -it --runtime=nvidia --gpus all --ipc=host \
  -v ~/cosmos3/Cosmos3-Nano:/workspace/cosmos3/Cosmos3-Nano:ro \
  -v ~/cosmos3/output:/workspace/cosmos3/output \
  cosmos3-nano:gb10 \
  bash -c "source .venv/bin/activate && python generate_video.py"
```

The model weights are mounted read-only — download them once on the host, use them in any container.

---

## What I Learned

**Unified memory changes the game for large model inference.** The GB10's 128GB shared pool means you can load models that would require multi-GPU sharding on discrete GPUs. No tensor parallelism, no FSDP, no distributed setup — just load and run.

**The Blackwell ecosystem is still maturing on ARM.** Flash Attention, NATTEN, and torch.compile all had issues. The SDPA fallback works but is slower than Flash Attention would be. As the ecosystem catches up, expect these patches to become unnecessary and performance to improve.

**Prompt upsampling is essential.** The quality gap between a raw text prompt and an LLM-upsampled structured prompt is not subtle — it's the difference between incoherent noise and a recognizable scene. Budget for the API call.

**480p is the practical ceiling on GB10.** The model supports 720p, but memory pressure makes it unreliable. 480p at 24fps produces good-looking video at a comfortable memory margin.

---

## What's Next

Cosmos3-Nano on GB10 is a starting point. The same model supports:

- **Action generation** — extracting robot trajectories from video (inverse dynamics) and predicting outcomes from action commands (forward dynamics)
- **Visual reasoning** — scene understanding with 256K token context via vLLM
- **Audio generation** — stereo 48kHz sound synchronized with video
- **Policy serving** — the Cosmos3-Nano-Policy-DROID variant runs as a real-time robot controller

And with LoRA fine-tuning on an RTX PRO 6000 (96GB), you can specialize the model for your specific domain — industrial scenes, specific robot platforms, custom environments.

But that's a story for another post.

---

## Resources

- **Cosmos3-Nano on HuggingFace:** https://huggingface.co/nvidia/Cosmos3-Nano
- **Cosmos3 Model Collection:** https://huggingface.co/collections/nvidia/cosmos3
- **Cosmos Framework:** https://github.com/NVIDIA/cosmos-framework
- **Cosmos Examples:** https://github.com/NVIDIA/cosmos
- **Project DIGITS:** https://www.nvidia.com/en-us/project-digits/
- **Technical Report:** https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf

---

*The complete setup guide, all patches, Docker containers, and interactive scripts are available in my project repository. The four GB10 patches can be auto-applied to a fresh cosmos-framework install with a single command: `python docker/patches/apply_patches.py`*
