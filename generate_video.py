"""Cosmos3-Nano video generator on GB10 — text-to-video and image-to-video.

Run and follow the prompts:
    source ~/cosmos3/activate.sh
    python ~/cosmos3/generate_video.py
"""
import json
import os
import time

import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image

MODEL = "/home/dgx-destro/cosmos3/Cosmos3-Nano"
NEGATIVE_PROMPT = "/home/dgx-destro/cosmos3/Cosmos3-Nano-assets/assets/negative_prompt.json"
OUTPUT_DIR = "/home/dgx-destro/cosmos3/output"

RESOLUTION_MAP = {
    256: {"16:9": (456, 256), "4:3": (336, 256), "1:1": (256, 256), "9:16": (256, 456)},
    480: {"16:9": (848, 480), "4:3": (640, 480), "1:1": (480, 480), "9:16": (480, 848)},
    720: {"16:9": (1280, 720), "4:3": (960, 720), "1:1": (720, 720), "9:16": (720, 1280)},
}


def build_prompt(text, duration, fps, resolution, aspect_ratio):
    w, h = RESOLUTION_MAP[resolution][aspect_ratio]
    return {
        "subjects": [{"description": text, "action": text}],
        "background_setting": "",
        "lighting": {"conditions": "natural lighting"},
        "aesthetics": {"composition": "well-framed", "color_scheme": "natural colors"},
        "cinematography": {
            "camera_motion": "steady",
            "framing": "centered on the action",
            "camera_angle": "eye level",
        },
        "context": text,
        "actions": [{"time": f"0:00-0:{duration:02.0f}", "description": text}],
        "segments": [{
            "segment_index": 0,
            "time_range": f"0:00-0:{duration:02.0f}",
            "description": text,
        }],
        "temporal_caption": text,
        "output_parameters": {
            "height": h,
            "width": w,
            "num_frames": round(duration * fps) + 1,
            "fps": fps,
            "resolution": f"{resolution}p",
            "aspect_ratio": aspect_ratio.replace(":", ","),
        },
    }


def get_input(prompt_text, default=None):
    if default:
        val = input(f"  {prompt_text} [{default}]: ").strip()
        return val if val else default
    return input(f"  {prompt_text}: ").strip()


def load_pipeline():
    print("  Loading Cosmos3-Nano pipeline...")
    pipe = Cosmos3OmniPipeline.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        enable_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=10.0
    )
    return pipe


def generate_one(pipe, negative_prompt):
    print()
    print("  Modes:")
    print("    1. Text-to-Video   (generate video from text prompt)")
    print("    2. Image-to-Video  (animate an image with text prompt)")
    print()
    mode = get_input("Select mode (1 or 2)", "1")

    image = None
    image_path = None
    if mode == "2":
        image_path = get_input("Image path")
        while not os.path.isfile(os.path.expanduser(image_path)):
            print(f"    File not found: {image_path}")
            image_path = get_input("Image path")
        image_path = os.path.abspath(os.path.expanduser(image_path))
        image = load_image(image_path)

    if mode == "2":
        prompt = get_input("Prompt (describe the motion/action)")
    else:
        prompt = get_input("Prompt (describe the scene and action)")

    duration = float(get_input("Duration in seconds", "3"))
    fps = int(get_input("FPS", "24"))
    resolution = int(get_input("Resolution (256/480/720)", "480"))
    aspect_ratio = get_input("Aspect ratio (16:9 / 4:3 / 1:1 / 9:16)", "16:9")
    steps = int(get_input("Inference steps", "35"))
    guidance = float(get_input("Guidance scale", "6.0"))
    seed = int(get_input("Seed", "42"))
    output_name = get_input("Output filename (leave blank for auto)", "")

    num_frames = round(duration * fps) + 1
    w, h = RESOLUTION_MAP[resolution][aspect_ratio]
    mode_tag = "i2v" if mode == "2" else "t2v"

    if output_name:
        out_path = output_name if os.path.isabs(output_name) else os.path.join(OUTPUT_DIR, output_name)
        if not out_path.endswith(".mp4"):
            out_path += ".mp4"
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(OUTPUT_DIR, f"{mode_tag}_{timestamp}.mp4")

    print()
    print("-" * 60)
    print(f"  Mode:       {'Image-to-Video' if mode == '2' else 'Text-to-Video'}")
    if image_path:
        print(f"  Image:      {image_path}")
    print(f"  Prompt:     {prompt}")
    print(f"  Duration:   {duration}s ({num_frames} frames @ {fps}fps)")
    print(f"  Resolution: {w}x{h} ({resolution}p {aspect_ratio})")
    print(f"  Steps:      {steps}")
    print(f"  Guidance:   {guidance}")
    print(f"  Seed:       {seed}")
    print(f"  Output:     {out_path}")
    print("-" * 60)
    print()

    json_prompt = build_prompt(prompt, duration, fps, resolution, aspect_ratio)

    pipe_kwargs = dict(
        prompt=json.dumps(json_prompt),
        negative_prompt=json.dumps(negative_prompt),
        num_frames=num_frames,
        height=h,
        width=w,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    )
    if image is not None:
        pipe_kwargs["image"] = image

    print("  Generating video...")
    t0 = time.time()
    result = pipe(**pipe_kwargs)
    elapsed = time.time() - t0

    export_to_video(result.video, out_path, fps=fps)

    print()
    print("=" * 60)
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Saved to {out_path}")
    print("=" * 60)

    return seed


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print()
    print("=" * 60)
    print("  Cosmos3-Nano  |  Video Generator  |  GB10")
    print("=" * 60)

    pipe = load_pipeline()
    negative_prompt = json.load(open(NEGATIVE_PROMPT))

    seed = generate_one(pipe, negative_prompt)

    while True:
        print()
        again = get_input("Generate another video? (y/n)", "n")
        if again.lower() not in ("y", "yes"):
            break
        seed = generate_one(pipe, negative_prompt)

    print("\n  Goodbye.")


if __name__ == "__main__":
    main()
