# Cosmos3-Nano-Policy-DROID — Real-Time Robot Policy Server

## What it is

A **closed-loop robot policy model** built on Cosmos3-Nano (16B). Given a language instruction and live camera observation, it generates the next action chunk for the robot to execute — in real-time, continuously.

Unlike base Cosmos3-Nano (single-shot video/trajectory generation), this model runs as a **WebSocket server** that streams actions to a robot client in a loop: observe → predict → act → repeat.

**Ranked #1 on RoboArena Policy Leaderboard.**

## Base Cosmos3-Nano vs Policy-DROID

| | Cosmos3-Nano | Policy-DROID |
|---|---|---|
| **Purpose** | Video gen + trajectory extraction | Real-time robot control |
| **Input** | Text/image/video | Language instruction + live camera |
| **Output** | Video or action trajectory | Action chunks streamed to robot |
| **Inference** | Single-shot (~20s for 30 steps) | Closed-loop (4 steps, ~1-2s) |
| **Action dim** | Varies by embodiment | 8D (joint_pos) or 10D (EE pose) |
| **Chunk size** | 16 frames | 32 steps |
| **Control FPS** | 10 | 15 Hz |
| **Training data** | General (1.3B data points) | DROID manipulation dataset |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    CONTROL LOOP                          │
│                                                          │
│   Language: "Pick up the banana"                         │
│        +                                                 │
│   Camera image (540×640 RGB)      ──►  Cosmos3-Nano      │
│        +                               Policy-DROID      │
│   Joint state [7D] + gripper [1D]      (16B MoT)        │
│                                           │              │
│                                    4-step denoise        │
│                                           │              │
│                                    Action chunk          │
│                                    [32 steps × 8D]       │
│                                           │              │
│                                    Robot executes        │
│                                           │              │
│                                    New observation ──┐   │
│                                                      │   │
│                              ◄───────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

## Server-Client Communication

```
┌─────────────────────────┐     WebSocket      ┌──────────────────────┐
│  Cosmos3 Policy Server  │◄──────────────────►│  Robot / Simulation  │
│                         │                     │                      │
│  Loads 16B model        │  observation dict   │  Isaac Sim (RoboLab) │
│  Receives camera image  │  ◄────────────────  │  or real robot       │
│  Runs 4-step denoise    │                     │                      │
│  Returns action chunk   │  action dict        │  Sends camera image  │
│                         │  ────────────────►  │  + joint state       │
│  Port 8000 (WebSocket)  │                     │  Executes actions    │
└─────────────────────────┘                     └──────────────────────┘

Protocol: OpenPI msgpack + NumPy over WebSocket
```

---

## Server Defaults

| Parameter | Value | Flag |
|---|---|---|
| Model | `nvidia/Cosmos3-Nano-Policy-DROID` | `--checkpoint-path` |
| Port | 8000 | `--port` |
| Domain | `droid_lerobot` | `--domain-name` |
| Action space | `joint_pos` (8D) | `--action-space` |
| Action dim | 8 | `--action-dim` |
| Chunk size | 32 steps | `--action-chunk-size` |
| Denoising steps | 4 | `--num-steps` |
| Guidance scale | 3.0 | `--guidance` |
| Shift | 5.0 | `--shift` |
| Conditioning FPS | 15 Hz | `--conditioning-fps` |
| Image size | 540×640 | `--image-height` / `--image-width` |
| Sampler | UniPC | `--sampler` |
| Seed | 0 | `--seed` |
| Use state | True | `--use-state` |
| History length | 1 | `--history-length` |

---

## Action Space Formats

### joint_pos (8D) — default

| Index | Dimension | Description |
|---|---|---|
| 0-6 | Joint positions | 7 joint angles (radians) |
| 7 | Gripper | Open/close command |

### midtrain (10D) — end-effector pose

| Index | Dimension | Description |
|---|---|---|
| 0-2 | Position | XYZ end-effector position |
| 3-8 | Rotation | 6D rotation representation (rot6d) |
| 9 | Gripper | Open/close command |

---

## Observation Dict (Client → Server)

```python
{
    # Required
    "observation/image": np.ndarray,             # [H, W, 3] uint8 RGB

    # Required for joint_pos action space
    "observation/joint_position": np.ndarray,    # [7] or [T, 7] float32
    "observation/gripper_position": np.ndarray,  # scalar or [T, 1] float32

    # Required
    "language_instruction": "pick up the banana and place it in the bowl",
}
```

### Multi-view (RoboArena format)

```python
{
    "observation/wrist_image_left": np.ndarray,       # [H, W, 3] wrist camera
    "observation/exterior_image_1_left": np.ndarray,  # [H, W, 3] external cam 1
    "observation/exterior_image_2_left": np.ndarray,  # [H, W, 3] external cam 2
    "observation/joint_position": np.ndarray,
    "observation/gripper_position": np.ndarray,
    "language_instruction": "...",
}
```

The server auto-composes multi-view into a single concat image:
- Top row: wrist camera (full resolution)
- Bottom row: two external cameras (half resolution, side by side)

## Action Dict (Server → Client)

```python
{
    "action": np.ndarray,    # [32, 8] float32 — 32-step action chunk
    "video": np.ndarray,     # [T, H, W, 3] uint8 — optional predicted rollout
}
```

---

## Setup on GB10

### Step 1: Install dependencies

```bash
source ~/cosmos3/activate.sh

# Install OpenPI server (WebSocket protocol)
pip install openpi-server

# Or install from Physical Intelligence repo
# pip install "openpi @ git+https://github.com/Physical-Intelligence/openpi.git"
```

### Step 2: Download Policy-DROID model

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/Cosmos3-Nano-Policy-DROID',
                  local_dir='$HOME/cosmos3/Cosmos3-Nano-Policy-DROID')
"
```

### Step 3: Start the policy server

```bash
source ~/cosmos3/activate.sh
export TORCH_COMPILE_DISABLE=1

cd ~/cosmos3/cosmos-framework

python -m cosmos_framework.scripts.action_policy_server_robolab \
    --checkpoint-path ~/cosmos3/Cosmos3-Nano-Policy-DROID \
    --port 8000 \
    --domain-name droid_lerobot \
    --action-space joint_pos \
    --action-dim 8 \
    --num-steps 4 \
    --guidance 3.0 \
    --conditioning-fps 15 \
    --action-chunk-size 32
```

### Step 4: Test with a Python client

```python
import numpy as np

# Using OpenPI WebSocket client
from openpi.serving.websocket_policy_client import WebsocketPolicyClient

client = WebsocketPolicyClient(host="localhost", port=8000)

# Create a dummy observation
observation = {
    "observation/image": np.random.randint(0, 255, (540, 640, 3), dtype=np.uint8),
    "observation/joint_position": np.zeros(7, dtype=np.float32),
    "observation/gripper_position": np.array(0.0, dtype=np.float32),
    "language_instruction": "pick up the object",
}

# Get action
result = client.infer(observation)
action = result["action"]  # [32, 8] — 32-step action chunk
print(f"Action shape: {action.shape}")
print(f"First step: {action[0]}")
```

### Or with raw WebSocket

```python
import asyncio
import msgpack
import numpy as np
import websockets

async def get_action():
    uri = "ws://localhost:8000"
    async with websockets.connect(uri) as ws:
        # Receive metadata
        metadata = msgpack.unpackb(await ws.recv())
        print(f"Server metadata: {metadata}")

        # Send observation
        obs = {
            "observation/image": np.random.randint(0, 255, (540, 640, 3), dtype=np.uint8),
            "observation/joint_position": np.zeros(7, dtype=np.float32),
            "observation/gripper_position": np.float32(0.0),
            "language_instruction": "pick up the object",
        }

        # Pack with msgpack (numpy arrays as bytes)
        packed = msgpack.packb(obs, default=lambda x: x.tobytes() if isinstance(x, np.ndarray) else x)
        await ws.send(packed)

        # Receive action
        result = msgpack.unpackb(await ws.recv())
        action = np.frombuffer(result["action"], dtype=np.float32).reshape(-1, 8)
        print(f"Action: {action.shape}")
        print(f"First step: {action[0]}")

asyncio.run(get_action())
```

---

## RoboLab Integration (Simulation)

### Setup RoboLab

```bash
git clone https://github.com/NVlabs/RoboLab.git
cd RoboLab

# Build Docker
./docker/build_docker.sh latest

# Run Docker
./docker/run_docker.sh latest
```

### Run tasks against Cosmos3 policy server

```bash
# Single environment
python policies/cosmos3/run.py \
    --task BananaInBowlTask

# Multiple parallel environments (headless)
python policies/cosmos3/run.py \
    --task BananaInBowlTask \
    --num-envs 10 \
    --headless

# Custom server address
python policies/cosmos3/run.py \
    --task BananaInBowlTask \
    --policy-host <GB10_IP> \
    --policy-port 8000
```

### Available tasks (120+ in RoboLab-120 suite)

Pick-and-place, stacking, rearrangement, tool use, and more. Each task includes:
- Language instruction
- Automated success/failure detection
- Composable predicates for evaluation

---

## Docker Setup

### Server container

```bash
docker run --rm -it \
    --runtime=nvidia --gpus all --ipc=host \
    --net=host \
    -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    -e TORCH_COMPILE_DISABLE=1 \
    -e LD_LIBRARY_PATH="" \
    -v ~/cosmos3/Cosmos3-Nano-Policy-DROID:/workspace/model:ro \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    cosmos3-nano:gb10 \
    bash -c "source .venv/bin/activate && \
    pip install openpi-server && \
    cd cosmos-framework && \
    python -m cosmos_framework.scripts.action_policy_server_robolab \
        --checkpoint-path /workspace/model \
        --port 8000 \
        --num-steps 4"
```

---

## Adapting for SO-101

To use Policy-DROID architecture with your SO-101 robot, you would:

### 1. Adjust observation format

```python
observation = {
    "observation/image": camera_image,           # [480, 640, 3] uint8
    "observation/joint_position": joint_pos,     # [5] float32 (5 joints)
    "observation/gripper_position": gripper_pos, # scalar float32
    "language_instruction": "pick the part from the bin",
}
```

### 2. Modify server parameters

```bash
python -m cosmos_framework.scripts.action_policy_server_robolab \
    --checkpoint-path ~/cosmos3/Cosmos3-Nano-Policy-DROID \
    --domain-name lerobot-so101 \
    --action-dim 6 \
    --action-space joint_pos \
    --action-chunk-size 16 \
    --conditioning-fps 30 \
    --image-height 480 \
    --image-width 640
```

### 3. Fine-tune on SO-101 data (recommended)

The base Policy-DROID is trained on DROID robot data (Franka Panda).
For best results on SO-101, fine-tune with LoRA on your SO-101 teleoperation dataset.
See `COSMOS3_FINETUNE_GUIDE.md` for the full fine-tuning pipeline.

---

## Performance Estimates

| Platform | Steps | Latency per chunk | Control rate |
|---|---|---|---|
| H100 | 4 | ~100-200 ms | ~5-10 Hz |
| GB10 | 4 | ~1-2 sec | ~0.5-1 Hz |
| GB10 | 2 | ~0.5-1 sec | ~1-2 Hz |

GB10 is usable for slow manipulation tasks. For real-time control at 15 Hz, H100 or similar is needed.

---

## All Server CLI Flags

```
--checkpoint-path       Model path or HF repo name [nvidia/Cosmos3-Nano-Policy-DROID]
--port                  WebSocket port [8000]
--host                  Bind address [0.0.0.0]
--domain-name           Action domain [droid_lerobot]
--action-space          joint_pos (8D) or midtrain (10D) [joint_pos]
--action-dim            Raw action dimension [8]
--action-chunk-size     Steps per chunk [32]
--num-steps             Denoising steps [4]
--guidance              Guidance scale [3.0]
--shift                 UniPC shift [5.0]
--conditioning-fps      Control FPS [15]
--resolution            Transform resolution [480]
--image-height          Input image height [540]
--image-width           Input image width [640]
--seed                  RNG seed [0]
--deterministic-seed    Same seed every request [false]
--decode-video          Return predicted rollout video [false]
--sampler               unipc or edm [unipc]
--use-state             Include current state [true]
--history-length        State history rows to trim [1]
--hf-revision           HF model revision [main]
--allow-dcp-checkpoint  Allow DCP path instead of safetensors [false]
```

---

## Key Links

- Model: https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID
- RoboLab: https://github.com/NVlabs/RoboLab
- Cosmos Framework: https://github.com/NVIDIA/cosmos-framework
- OpenPI (WebSocket protocol): https://github.com/Physical-Intelligence/openpi
- Technical report: https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf
- RoboArena leaderboard: https://huggingface.co/spaces/nvidia/RoboArena
