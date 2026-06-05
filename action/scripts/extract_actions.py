"""Extract robot action trajectory from a video using Cosmos3-Nano inverse dynamics.

Requires vLLM-Omni server running on localhost:8000.
Start it with:
    ~/cosmos3/action/scripts/start_server.sh

Usage:
    python ~/cosmos3/action/scripts/extract_actions.py
"""
import json
import os
import time

import requests

SERVER_URL = "http://localhost:8000"
OUTPUT_DIR = "/home/dgx-destro/cosmos3/action/output"

EMBODIMENTS = {
    "ur":        {"raw_action_dim": 10, "desc": "UR robot (10D)"},
    "franka":    {"raw_action_dim": 10, "desc": "Franka Panda single arm + RobotiQ (10D)"},
    "franka_dual": {"raw_action_dim": 20, "desc": "Franka Panda dual arm (20D)"},
    "agibot":    {"raw_action_dim": 29, "desc": "Agibot (29D)"},
    "widowx":    {"raw_action_dim": 10, "desc": "WidowX 250 (10D)"},
    "google":    {"raw_action_dim": 10, "desc": "Google robot (10D)"},
    "umi":       {"raw_action_dim": 9,  "desc": "UMI (9D)"},
    "av":        {"raw_action_dim": 9,  "desc": "Autonomous vehicle (9D)"},
    "camera":    {"raw_action_dim": 9,  "desc": "General camera motion (9D)"},
}


def get_input(prompt_text, default=None):
    if default:
        val = input(f"{prompt_text} [{default}]: ").strip()
        return val if val else default
    return input(f"{prompt_text}: ").strip()


def check_server():
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  Cosmos3-Nano  |  Inverse Dynamics  |  Video → Actions")
    print("=" * 60)
    print()

    if not check_server():
        print("ERROR: vLLM-Omni server not running on localhost:8000")
        print("Start it with:  ~/cosmos3/start_server.sh")
        return

    print("Server is running.\n")

    video_path = get_input("Video path")
    while not os.path.isfile(os.path.expanduser(video_path)):
        print(f"  File not found: {video_path}")
        video_path = get_input("Video path")
    video_path = os.path.expanduser(video_path)

    print("\nAvailable robot embodiments:")
    for key, val in EMBODIMENTS.items():
        print(f"  {key:15s} → {val['desc']}")
    print()

    domain = get_input("Robot embodiment", "ur")
    while domain not in EMBODIMENTS:
        print(f"  Unknown: {domain}. Choose from: {', '.join(EMBODIMENTS.keys())}")
        domain = get_input("Robot embodiment", "ur")

    embodiment = EMBODIMENTS[domain]
    action_chunk_size = int(get_input("Action chunk size (frames per chunk)", "60"))
    prompt = get_input("Scene description", "A robot performing a manipulation task")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"actions_{domain}_{timestamp}.json")

    print()
    print("-" * 60)
    print(f"  Video:       {video_path}")
    print(f"  Embodiment:  {embodiment['desc']}")
    print(f"  Action dim:  {embodiment['raw_action_dim']}D")
    print(f"  Chunk size:  {action_chunk_size} frames")
    print(f"  Output:      {out_path}")
    print("-" * 60)
    print()

    extra_params = {
        "action_mode": "inverse_dynamics",
        "domain_name": domain,
        "action_chunk_size": action_chunk_size,
        "image_size": 480,
        "view_point": "ego_view",
        "raw_action_dim": embodiment["raw_action_dim"],
        "guardrails": False,
    }

    data = {
        "prompt": prompt,
        "num_frames": str(action_chunk_size + 1),
        "fps": "10",
        "num_inference_steps": "30",
        "guidance_scale": "1.0",
        "flow_shift": "10.0",
        "seed": "0",
        "extra_params": json.dumps(extra_params),
    }

    print("Submitting to server...")
    with open(video_path, "rb") as video_file:
        files = {
            "input_reference": (os.path.basename(video_path), video_file, "video/mp4"),
        }
        response = requests.post(
            f"{SERVER_URL}/v1/videos",
            data=data,
            files=files,
            timeout=60,
        )

    if response.status_code != 200:
        print(f"ERROR: Server returned {response.status_code}")
        print(response.text)
        return

    initial = response.json()
    task_id = initial.get("id")
    print(f"Task ID: {task_id}")
    print("Processing", end="", flush=True)

    while True:
        r = requests.get(f"{SERVER_URL}/v1/videos/{task_id}", timeout=30)
        result = r.json()
        status = result.get("status", "unknown")
        progress = result.get("progress", 0)

        if status == "completed":
            print(f"\rProcessing... done!{' ' * 20}")
            break
        elif status in ("failed", "cancelled"):
            print(f"\rERROR: Task {status}")
            print(json.dumps(result, indent=2))
            return
        else:
            print(f"\rProcessing... {progress}%", end="", flush=True)
            time.sleep(2)

    action = result.get("action")

    output_data = {
        "video": video_path,
        "embodiment": domain,
        "action_dim": embodiment["raw_action_dim"],
        "description": embodiment["desc"],
        "chunk_size": action_chunk_size,
        "prompt": prompt,
        "action": action,
    }

    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print()
    print("=" * 60)
    if action and isinstance(action, list):
        print(f"  Extracted {len(action)} action frames")
        print(f"  Each frame: {embodiment['raw_action_dim']}D vector")
        print(f"  First frame: {action[0][:5]}..." if len(action[0]) > 5 else f"  First frame: {action[0]}")
    print(f"  Saved to {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
