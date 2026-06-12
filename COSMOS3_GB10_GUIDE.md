# Cosmos3-Nano on GB10 — Complete Setup Guide

## System Info

| Component | Detail |
|---|---|
| **Model** | Cosmos3-Nano (16B params, 33GB on disk) |
| **GPU** | NVIDIA GB10 (Project DIGITS), Compute 12.1 |
| **Memory** | 128GB unified (CPU+GPU) |
| **CUDA** | 13.0, Driver 580.126.09 |
| **OS** | Ubuntu 24.04 (aarch64/ARM) |
| **Stack** | Python 3.13, PyTorch 2.12+cu130, Diffusers (from git), cosmos-framework 1.2.2 |
| **Location** | `~/cosmos3/` |
| **IP** | `192.168.1.25` |
| **User** | `dgx-destro` |

## Quick Start

```bash
source ~/cosmos3/activate.sh
export TORCH_COMPILE_DISABLE=1    # required for action generation
```

## Directory Layout

```
~/cosmos3/
├── .venv/                          # Python 3.13 + PyTorch 2.12 + CUDA 13.0
├── activate.sh                     # Quick env activation
├── cosmos-framework/               # NVIDIA framework (patched for GB10)
├── Cosmos3-Nano/                   # Model weights (33GB)
├── Cosmos3-Nano-assets/assets/     # Example prompts & sample inputs
├── output/                         # Generated output directory
├── webui.py                        # Gradio Web UI (NVIDIA themed)
├── generate_video.py               # Interactive image-to-video script
├── test_t2v.py                     # Text-to-video example
├── test_i2v.py                     # Image-to-video example
├── generate_image.py               # Image generation script
├── thor_test_t2v.py                # Thor text-to-video test
├── thor_test_i2v.py                # Thor image-to-video test
├── webui.sh                        # Docker Web UI shortcut
├── docker/                         # Dockerfiles for GB10 & Thor
│   ├── Dockerfile.gb10
│   ├── Dockerfile.jetson-thor
│   ├── Dockerfile.jetson-thor-vllm
│   ├── patches/
│   │   ├── apply_patches.py
│   │   └── sdpa_fallback.py
│   ├── build.sh
│   └── run.sh
├── COSMOS3_GB10_GUIDE.md           # This file
└── COSMOS3_JETSON_THOR_GUIDE.md    # Thor setup guide
```

---

## GB10 Patches Applied to cosmos-framework

The framework needed 4 patches for GB10 compatibility. These are already applied in `~/cosmos3/cosmos-framework/`:

| # | File | Issue | Fix |
|---|---|---|---|
| 1 | `inference/args.py` | `nvmlDeviceGetMemoryInfo` not supported on GB10 | Fallback to `torch.cuda.get_device_properties(0).total_mem` |
| 2 | `inference/vision.py` | `torchvision.io.read_video` removed in v0.27 | Replaced with PyAV (`av` library) |
| 3 | `model/attention/sdpa_fallback.py` (new) | No flash-attn/natten on GB10 | PyTorch native SDPA with GQA head expansion |
| 4 | `model/attention/frontend.py` + `backends.py` | SDPA not registered as backend | Added `"sdpa"` to `BACKEND_MAP` and `BACKEND_CHECK_MAP` |

### Required Environment Variables

```bash
export TORCH_COMPILE_DISABLE=1                    # torch.compile fails on GB10 attention
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"  # prevents OOM fragmentation
export LD_LIBRARY_PATH=                            # must be cleared to avoid library conflicts
```

---

## Web UI (Gradio)

The web UI provides a browser-based interface for all generation modes with NVIDIA-themed styling.

### Launch (Bare Metal)

```bash
source ~/cosmos3/activate.sh
pip install gradio    # first time only
python webui.py
```

Access at: `http://192.168.1.25:7860` (from network) or `http://localhost:7860` (local).

### Launch (Docker)

```bash
docker/run.sh gb10 webui           # default port 7860
docker/run.sh gb10 webui 7861      # custom port
./webui.sh                          # shortcut (port 7860)
./webui.sh 7861                     # shortcut (custom port)
```

### To change host/port

```bash
WEBUI_HOST=0.0.0.0 WEBUI_PORT=8080 python webui.py
```

### Web UI Features

| Tab | Input | Output |
|---|---|---|
| **Text to Video** | Text prompt | MP4 video preview + download |
| **Image to Video** | Drag & drop image + optional text | MP4 video preview + download |
| **Text to Image** | Text prompt | JPG image preview + download |

Controls per tab: resolution (256p/480p/720p), aspect ratio, frames, FPS, inference steps, guidance scale, seed.

### Web UI Technical Details

| Feature | Detail |
|---|---|
| **NVIDIA Theme** | Custom dark theme with NVIDIA Green (#76B900), gradient header with logo, green tabs/buttons. Theme and CSS passed to `launch()` (Gradio 6.x requirement) |
| **Live Status** | Pipeline runs in background thread; main thread yields `"Generating... Xs"` status every 3 seconds to keep SSE connection alive |
| **Queuing** | `app.queue(default_concurrency_limit=1)` for long-running task support |
| **H.264 Re-encoding** | ffmpeg re-encodes to `libx264 + yuv420p + faststart` for browser playback |
| **ffmpeg Fallback** | If re-encode fails, logs error to terminal and falls back to raw video |
| **Video Component** | `gr.Video(autoplay=True)` — do NOT use `format="mp4"` (breaks display in Gradio 6.x) |
| **Output Path** | Videos saved to `~/cosmos3/output/` (in Gradio `allowed_paths`) |
| **Pipeline Caching** | Model loads once on first request; subsequent generations reuse loaded pipeline |

### Web UI Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| "Connection lost" / page reload | SSE timeout during ~3 min generation | Fixed — pipeline runs in background thread, yields keepalive every 3s |
| Video not showing in preview | `format="mp4"` in `gr.Video` breaks Gradio 6.x | Fixed — removed `format="mp4"`, use `gr.Video(autoplay=True)` only |
| `Cosmos3OmniPipeline` ignores `callback_on_step_end` | Pipeline doesn't support step callbacks | Fixed — use background thread + `yield gr.skip()` instead |
| Gradio 6.x warning about theme/css | Params moved from `Blocks()` to `launch()` | Fixed — theme/css passed to `app.launch()` |
| Browser console: "Method not implemented" / "Empty string passed to getElementById()" | Gradio 6.x bundled JS + Firefox autofill scanning form inputs | Harmless — does not affect functionality. Ignore these warnings |
| ffmpeg not found | Missing system package | `sudo apt install ffmpeg` |
| Blank preview but download works | ffmpeg codec issue | Check terminal for `[ffmpeg] FAILED` — install libx264 codec |

---

## 1. Text-to-Video (Diffusers)

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

MODEL = "/home/dgx-destro/cosmos3/Cosmos3-Nano"

pipe = Cosmos3OmniPipeline.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    enable_safety_checker=False,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0
)

json_prompt = json.load(open("your_prompt.json"))
negative_prompt = json.load(open("/home/dgx-destro/cosmos3/Cosmos3-Nano-assets/assets/negative_prompt.json"))

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    num_frames=57,          # 57 = 2.4s @ 24fps
    height=480,             # 480p recommended for GB10
    width=848,              # 16:9 at 480p
    num_inference_steps=35,
    guidance_scale=6.0,
    generator=torch.Generator(device="cuda").manual_seed(42),
)

export_to_video(result.video, "output.mp4", fps=24)
```

## 2. Image-to-Video (Interactive)

```bash
source ~/cosmos3/activate.sh
python ~/cosmos3/generate_video.py
```

Prompts for image path, text description, duration, resolution, etc. All settings have defaults.

### Or programmatically:

```python
from diffusers.utils import load_image

image = load_image("your_input_image.jpg")

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    image=image,
    num_frames=73,          # 73 = 3s @ 24fps
    height=480,
    width=848,
    num_inference_steps=35,
    guidance_scale=6.0,
    generator=torch.Generator(device="cuda").manual_seed(42),
)
```

## 3. Prompt Upsampling (via Claude API)

Converts simple text into rich structured JSON prompts for better generation quality.

```bash
source ~/cosmos3/.venv/bin/activate

export PROMPT_UPSAMPLER_ENDPOINT_URL="https://api.anthropic.com/v1/"
export PROMPT_UPSAMPLER_MODEL_NAME="claude-sonnet-4-6"
export PROMPT_UPSAMPLER_API_TOKEN="<your_anthropic_api_key>"

echo "A robotic gripper picking metal parts from a bin" > /tmp/my_prompt.txt

python -m cosmos_framework.inference.prompt_upsampling \
    --input /tmp/my_prompt.txt \
    --output /tmp/upsampled/ \
    --mode text2video \
    --endpoint-url "${PROMPT_UPSAMPLER_ENDPOINT_URL}" \
    --model "${PROMPT_UPSAMPLER_MODEL_NAME}" \
    --api-token "${PROMPT_UPSAMPLER_API_TOKEN}" \
    --resolution 480 \
    --aspect-ratio "16,9"
```

For image-to-video: `--mode image2video --image-list images.txt`

---

## 4. Action Generation — Inverse Dynamics (Video to Trajectory)

Extracts robot action trajectory data from a video using cosmos-framework.

### Step 1: Create input JSON

```json
{
  "action_chunk_size": 16,
  "domain_name": "robomind-ur",
  "fps": 10,
  "image_size": 480,
  "view_point": "ego_view",
  "model_mode": "inverse_dynamics",
  "name": "my_inverse_test",
  "prompt": "A robot arm performing a manipulation task",
  "seed": 0,
  "vision_path": "/path/to/your/video.mp4"
}
```

### Step 2: Run

```bash
source ~/cosmos3/activate.sh
export TORCH_COMPILE_DISABLE=1
cd ~/cosmos3/cosmos-framework

python -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i ~/cosmos3/inputs/your_input.json \
    -o ~/cosmos3/output/action_output \
    --checkpoint-path ~/cosmos3/Cosmos3-Nano \
    --no-guardrails \
    --seed=0
```

### Step 3: Read output

Output at `~/cosmos3/output/action_output/<name>/sample_outputs.json`:

```python
import json
data = json.load(open("sample_outputs.json"))
action = data["outputs"][0]["content"]["action"]
# action = [[0.06, -0.07, 0.09, ...], [0.04, -0.07, 0.08, ...], ...]
# Each frame: 10D vector (for UR robot)
print(f"Frames: {len(action)}, Dims: {len(action[0])}")
```

### Output Format (UR robot 10D example)

```
Frame  0: [0.0608, -0.0663, 0.0943, 0.9527, -0.1202, -0.1616, 0.0797, 0.9937, 0.0573, 0.1766]
Frame  1: [0.0407, -0.0693, 0.0832, 0.9585, -0.1206, -0.1554, 0.1148, 0.9803, 0.0539, 0.1791]
...
Frame 15: [0.0965, -0.0610, 0.1674, 0.9784, -0.0655, -0.1163, 0.0607, 0.9859, 0.0301, 0.4142]
```

## 5. Action Generation — Forward Dynamics (Trajectory to Video)

Predicts what a robot will look like executing a given action sequence.

### Input JSON

```json
{
  "action_chunk_size": 16,
  "domain_name": "robomind-ur",
  "fps": 10,
  "image_size": 480,
  "view_point": "ego_view",
  "model_mode": "forward_dynamics",
  "name": "my_forward_test",
  "prompt": "A robot arm picking a part from a bin",
  "seed": 0,
  "vision_path": "/path/to/first_frame.png",
  "action_path": "/path/to/actions.json"
}
```

Where `actions.json` is a 2D array: `[[v1, v2, ..., v10], [v1, v2, ..., v10], ...]`

### Run (same command)

```bash
python -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i ~/cosmos3/inputs/forward_input.json \
    -o ~/cosmos3/output/forward_output \
    --checkpoint-path ~/cosmos3/Cosmos3-Nano \
    --no-guardrails \
    --seed=0
```

Output includes predicted video at `<name>/vision.mp4`.

---

## Supported Robot Embodiments

| Domain Name | Dim | Robot |
|---|---|---|
| `lerobot-so101` | 6D | **LeRobot SO-101 (5 joints + gripper)** (custom registered) |
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
| `hand_pose` | 57D | Egocentric hand |
| `pusht` | 2D | Push-T |

### SO-101 Action Vector Layout

```
Index 0: shoulder_pan.pos    (joint 1 — base rotation)
Index 1: shoulder_lift.pos   (joint 2 — shoulder up/down)
Index 2: elbow_flex.pos      (joint 3 — elbow bend)
Index 3: wrist_flex.pos      (joint 4 — wrist pitch)
Index 4: wrist_roll.pos      (joint 5 — wrist rotation)
Index 5: gripper.pos         (gripper open/close)
```

Registered in: `cosmos-framework/cosmos_framework/data/vfm/action/domain_utils.py`
- Domain ID: 16
- Action dim: 6

### Adding a New Robot Embodiment

Edit `~/cosmos3/cosmos-framework/cosmos_framework/data/vfm/action/domain_utils.py`:

```python
# Add to EMBODIMENT_TO_DOMAIN_ID:
"my-robot-name": <unused_id>,     # pick unused: 9-11, 17-19, etc.

# Add to EMBODIMENT_TO_RAW_ACTION_DIM:
"my-robot-name": <num_dims>,      # your action vector size
```

Then add to `~/cosmos3/extract_trajectory.py` EMBODIMENTS dict:

```python
"my-robot-name": {"dim": <num_dims>, "desc": "My Robot Description"},
```

---

## LeRobot Dataset Format (for SO-101 fine-tuning)

### Sample Dataset

`sreetz-nv/so101_teleop_vials_rack_left` — SO-101 teleoperation, vials rack task.

| Property | Value |
|---|---|
| Robot | so101_follower |
| Episodes | 75 |
| Total frames | 18,250 |
| FPS | 30 Hz |
| Action dim | 6D |
| State dim | 6D |
| Cameras | ego + Intel D455 external (480x640, AV1) |
| LeRobot version | v3.0 |
| Size | ~300 MB |

Downloaded to: `~/cosmos3/datasets/so101_vials/`

### LeRobot v3.0 Dataset Structure

```
my_dataset/
├── meta/
│   ├── info.json                       # Dataset metadata, features, fps
│   ├── stats.json                      # Action/state normalization stats
│   ├── tasks.parquet                   # Task descriptions
│   └── episodes/chunk-000/            # Per-episode metadata parquets
├── data/
│   └── chunk-000/
│       ├── file-000.parquet            # Action/state/timestamp per frame
│       ├── file-001.parquet
│       └── ...
└── videos/
    ├── observation.images.ego/
    │   └── chunk-000/
    │       ├── file-000.mp4
    │       └── ...
    └── observation.images.external_D455/
        └── chunk-000/
            └── ...
```

### info.json Key Fields

```json
{
  "codebase_version": "v3.0",
  "robot_type": "so101_follower",
  "total_episodes": 75,
  "total_frames": 18250,
  "fps": 30,
  "features": {
    "observation.state": {
      "dtype": "float32",
      "shape": [6],
      "names": ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
    },
    "action": {
      "dtype": "float32",
      "shape": [6],
      "names": ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
    },
    "observation.images.ego": {
      "dtype": "video",
      "shape": [480, 640, 3],
      "info": {"video.fps": 30, "video.codec": "av1"}
    }
  }
}
```

### Parquet Columns (per frame)

| Column | Type | Description |
|---|---|---|
| `action` | float32[6] | Joint positions + gripper command |
| `observation.state` | float32[6] | Current joint positions + gripper |
| `timestamp` | float32 | Time in seconds |
| `frame_index` | int64 | Frame within episode |
| `episode_index` | int64 | Episode number |
| `task_index` | int64 | Task ID |
| `index` | int64 | Global row index |

### Recording Your Own LeRobot Dataset

```bash
# Install lerobot
pip install lerobot

# Record teleoperation episodes
python -m lerobot.scripts.control_robot \
    --robot.type=so101_follower \
    --control.type=teleoperate \
    --control.fps=30 \
    --control.dataset_repo_id=your-hf-user/my_so101_dataset \
    --control.num_episodes=50
```

### Loading a LeRobot Dataset

```python
import pandas as pd
import numpy as np

# Read a single episode
df = pd.read_parquet("~/cosmos3/datasets/so101_vials/data/chunk-000/file-000.parquet")

# Extract actions
actions = np.stack(df["action"].values)
print(f"Episode shape: {actions.shape}")  # (N_frames, 6)

# Action stats
labels = ["sh_pan", "sh_lift", "elb_flx", "wr_flx", "wr_roll", "gripper"]
for i, l in enumerate(labels):
    print(f"  {l}: min={actions[:,i].min():.2f} max={actions[:,i].max():.2f}")
```

---

## Docker Setup

### Build

```bash
docker/build.sh gb10               # GB10
docker/build.sh jetson-thor         # Jetson Thor
docker/build.sh thor-vllm           # Thor + vLLM
```

### Run

```bash
# Interactive shell
docker/run.sh gb10 interactive

# Web UI
docker/run.sh gb10 webui            # port 7860
docker/run.sh gb10 webui 7861       # custom port
./webui.sh                           # shortcut

# Text-to-Video test
docker/run.sh gb10 t2v

# Image-to-Video generator
docker/run.sh gb10 generate
```

Access Web UI from network: `http://192.168.1.25:7860`

All modes work with any platform: `gb10`, `jetson-thor`, `thor-vllm`.

---

## GB10 Performance

| Task | Steps | Speed | Total Time |
|---|---|---|---|
| Text-to-video (480p, 57 frames) | 35 | ~5.3 s/step | ~3 min |
| Image-to-video (480p, 57 frames) | 35 | ~5.3 s/step | ~3 min |
| Inverse dynamics (16 frames) | 30 | ~1.46 it/s | ~20 sec |
| Model load (first time) | - | - | ~5 sec |
| Model load (cached) | - | - | ~3 sec |

### If you hit OOM

1. Reduce resolution: `height=256, width=456`
2. Reduce frames: `num_frames=25`
3. Reduce inference steps: `num_inference_steps=20`
4. Use `--enable-layerwise-offload` in vLLM mode

### To try 720p

```python
result = pipe(
    ...,
    num_frames=57,
    height=720,
    width=1280,
    num_inference_steps=25,
)
```

---

## Supported Resolutions & Aspect Ratios

| Resolution | 16:9 | 4:3 | 1:1 | 3:4 | 9:16 |
|---|---|---|---|---|---|
| 256p | 456x256 | 336x256 | 256x256 | 256x336 | 256x456 |
| 480p | 848x480 | 640x480 | 480x480 | 480x640 | 480x848 |
| 720p | 1280x720 | 960x720 | 720x720 | 720x960 | 720x1280 |

---

## vLLM-Omni Server (Docker)

For serving generation via REST API. Requires `nvidia/Cosmos-1.0-Guardrail` access.

```bash
# Request access first: https://huggingface.co/nvidia/Cosmos-1.0-Guardrail
~/cosmos3/start_server.sh
```

Then query at `http://localhost:8000/v1/videos/sync`.

## vLLM Reasoner Server

For visual understanding / reasoning tasks:

```bash
source ~/cosmos3/activate.sh

CUDA_VISIBLE_DEVICES=0 \
vllm serve ~/cosmos3/Cosmos3-Nano \
  --hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}' \
  --tensor-parallel-size 1 \
  --mm-encoder-tp-mode data \
  --async-scheduling \
  --allowed-local-media-path / \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --port 8000
```

```python
import openai

client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "file:///path/to/image.png"}},
            {"type": "text", "text": "Describe what you see."},
        ],
    }],
    max_tokens=1024,
)
print(response.choices[0].message.content)
```

---

## Common Dependency Issues

| Error | Fix |
|---|---|
| `cannot import name 'Cosmos3OmniPipeline'` | `pip install "diffusers @ git+https://github.com/huggingface/diffusers.git"` |
| `huggingface-hub>=0.34.0,<1.0 is required` | `pip install transformers -U` |
| `No module named 'peft'` | `pip install peft` |
| `cosmos-framework requires transformers<5.0.0` | Install cosmos-framework first, then diffusers from git last |
| `PermissionError: '/workspace'` | Set `export COSMOS3_DIR=~/cosmos3` |

### Recommended Install Order

```bash
# 1. cosmos-framework (pins transformers<5, diffusers)
cd ~/cosmos3/cosmos-framework
pip install -e ".[guardrail]"

# 2. peft (required by cosmos_guardrail but not declared)
pip install peft

# 3. diffusers from git (overrides framework's older pin — must be last)
pip install "diffusers @ git+https://github.com/huggingface/diffusers.git"
```

---

## Safety Checker

The safety checker requires access to the gated `nvidia/Cosmos-1.0-Guardrail` model.
Request access at: https://huggingface.co/nvidia/Cosmos-1.0-Guardrail

Once approved, set `enable_safety_checker=True` in the diffusers pipeline.

## Dependency Notes

- `diffusers` must be installed from git for `Cosmos3OmniPipeline`
- `lerobot` install conflicts with torch/diffusers versions — ignore pip compatibility warnings
- Qwen3-VL-8B-Instruct tokenizer (~17GB) is cached at `~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/`
- Wan2.2 VAE cached at `~/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B/`

## Network Info

| Machine | IP | User |
|---|---|---|
| GB10 (Project DIGITS) | 192.168.1.25 | dgx-destro |
| Jetson Thor | 192.168.1.29 | nvidia-thor |

### Copy files between machines

```bash
# GB10 to Thor:
scp /home/dgx-destro/cosmos3/<file> nvidia-thor@192.168.1.29:/home/nvidia-thor/cosmos3/<file>

# Thor to GB10:
scp nvidia-thor@192.168.1.29:/home/nvidia-thor/cosmos3/<file> /home/dgx-destro/cosmos3/<file>
```

## Key Links

- Model: https://huggingface.co/nvidia/Cosmos3-Nano
- Model collection: https://huggingface.co/collections/nvidia/cosmos3
- Framework: https://github.com/NVIDIA/cosmos-framework
- Examples: https://github.com/NVIDIA/cosmos
- Technical report: https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf
- Guardrail access: https://huggingface.co/nvidia/Cosmos-1.0-Guardrail
