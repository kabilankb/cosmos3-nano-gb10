"""Cosmos3-Nano inverse dynamics — extract robot action trajectory from video.

Run and follow the prompts:
    source ~/cosmos3/activate.sh
    export TORCH_COMPILE_DISABLE=1
    python ~/cosmos3/action/scripts/extract_trajectory.py
"""
import json
import os
import subprocess
import sys
import time

COSMOS_FRAMEWORK = "/home/dgx-destro/cosmos3/cosmos-framework"
CHECKPOINT = "/home/dgx-destro/cosmos3/Cosmos3-Nano"
INPUT_DIR = "/home/dgx-destro/cosmos3/action/inputs"
OUTPUT_DIR = "/home/dgx-destro/cosmos3/action/output"

EMBODIMENTS = {
    "robomind-ur":          {"dim": 10, "desc": "UR robot"},
    "robomind-franka":      {"dim": 10, "desc": "Franka Panda + RobotiQ"},
    "robomind-franka-dual": {"dim": 20, "desc": "Dual Franka Panda"},
    "bridge_orig_lerobot":  {"dim": 10, "desc": "WidowX 250"},
    "droid_lerobot":        {"dim": 10, "desc": "DROID"},
    "agibotworld":          {"dim": 29, "desc": "Agibot"},
    "umi":                  {"dim": 10, "desc": "UMI"},
    "fractal":              {"dim": 10, "desc": "Google robot"},
    "av":                   {"dim":  9, "desc": "Autonomous vehicle"},
    "camera_pose":          {"dim":  9, "desc": "Camera motion"},
    "hand_pose":            {"dim": 57, "desc": "Egocentric hand"},
    "pusht":                {"dim":  2, "desc": "Push-T"},
    "lerobot-so101":        {"dim":  6, "desc": "LeRobot SO-101 (5 joints + gripper)"},
}


def get_input(prompt_text, default=None):
    if default:
        val = input(f"  {prompt_text} [{default}]: ").strip()
        return val if val else default
    return input(f"  {prompt_text}: ").strip()


def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.environ.get("TORCH_COMPILE_DISABLE"):
        os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["LD_LIBRARY_PATH"] = ""

    print()
    print("=" * 65)
    print("  Cosmos3-Nano  |  Inverse Dynamics  |  Video → Trajectory")
    print("=" * 65)
    print()

    video_path = get_input("Video path")
    while not os.path.isfile(os.path.expanduser(video_path)):
        print(f"    File not found: {video_path}")
        video_path = get_input("Video path")
    video_path = os.path.abspath(os.path.expanduser(video_path))

    print()
    print("  Available robot embodiments:")
    for key, val in EMBODIMENTS.items():
        print(f"    {key:25s} {val['dim']:3d}D   {val['desc']}")
    print()

    domain = get_input("Robot embodiment", "robomind-ur")
    while domain not in EMBODIMENTS:
        print(f"    Unknown: {domain}")
        domain = get_input("Robot embodiment", "robomind-ur")

    chunk_size = int(get_input("Action chunk size (frames)", "16"))
    fps = int(get_input("FPS", "10"))
    prompt = get_input("Scene description", "A robot arm performing a manipulation task")
    seed = int(get_input("Seed", "0"))
    run_name = get_input("Run name", f"inverse_{int(time.time())}")

    embodiment = EMBODIMENTS[domain]
    input_json_path = os.path.join(INPUT_DIR, f"{run_name}.json")
    output_dir = os.path.join(OUTPUT_DIR, f"{run_name}_output")

    input_config = {
        "action_chunk_size": chunk_size,
        "domain_name": domain,
        "fps": fps,
        "image_size": 480,
        "view_point": "ego_view",
        "model_mode": "inverse_dynamics",
        "name": run_name,
        "prompt": prompt,
        "seed": seed,
        "vision_path": video_path,
    }

    with open(input_json_path, "w") as f:
        json.dump(input_config, f, indent=2)

    print()
    print("-" * 65)
    print(f"  Video:        {video_path}")
    print(f"  Robot:        {embodiment['desc']} ({domain}, {embodiment['dim']}D)")
    print(f"  Chunk size:   {chunk_size} frames @ {fps} fps")
    print(f"  Prompt:       {prompt}")
    print(f"  Seed:         {seed}")
    print(f"  Input JSON:   {input_json_path}")
    print(f"  Output dir:   {output_dir}")
    print("-" * 65)
    print()

    confirm = input("  Proceed? [Y/n]: ").strip().lower()
    if confirm in ("n", "no"):
        print("  Cancelled.")
        return

    print()
    print("  Running inverse dynamics...")
    print()

    t0 = time.time()
    result = subprocess.run(
        [
            sys.executable, "-m", "cosmos_framework.scripts.inference",
            "--parallelism-preset=latency",
            "-i", input_json_path,
            "-o", output_dir,
            "--checkpoint-path", CHECKPOINT,
            "--no-guardrails",
            "--seed", str(seed),
        ],
        cwd=COSMOS_FRAMEWORK,
        capture_output=False,
    )

    if result.returncode != 0:
        print(f"\n  ERROR: Inference failed (exit code {result.returncode})")
        return

    elapsed = time.time() - t0

    output_json = os.path.join(output_dir, run_name, "sample_outputs.json")
    if not os.path.isfile(output_json):
        print(f"\n  ERROR: Output file not found at {output_json}")
        return

    data = json.load(open(output_json))
    action = data["outputs"][0]["content"]["action"]

    print()
    print("=" * 65)
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Frames: {len(action)}, Dims per frame: {len(action[0])}")
    print("=" * 65)
    print()

    header = f"{'Frame':>7} |"
    for d in range(len(action[0])):
        header += f" {'D'+str(d):>8}"
    print(header)
    print("-" * (10 + 9 * len(action[0])))

    for i, frame in enumerate(action):
        row = f"{i:>7} |"
        for v in frame:
            row += f" {v:>8.4f}"
        print(row)

    traj_path = os.path.join(output_dir, f"{run_name}_trajectory.json")
    traj_data = {
        "video": video_path,
        "domain": domain,
        "robot": embodiment["desc"],
        "action_dim": embodiment["dim"],
        "chunk_size": chunk_size,
        "fps": fps,
        "prompt": prompt,
        "seed": seed,
        "num_frames": len(action),
        "trajectory": action,
    }
    with open(traj_path, "w") as f:
        json.dump(traj_data, f, indent=2)

    print()
    print(f"  Trajectory saved to: {traj_path}")
    print(f"  Full output at:      {output_json}")
    print()


if __name__ == "__main__":
    main()
