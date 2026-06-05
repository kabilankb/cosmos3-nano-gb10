"""Cosmos3-Nano text-to-image generator on GB10.

Run and follow the prompts:
    source ~/cosmos3/activate.sh
    python ~/cosmos3/generate_image.py
"""
import json
import os
import time

import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

MODEL = "/home/dgx-destro/cosmos3/Cosmos3-Nano"
NEGATIVE_PROMPT = "/home/dgx-destro/cosmos3/Cosmos3-Nano-assets/assets/negative_prompt.json"
OUTPUT_DIR = "/home/dgx-destro/cosmos3/output"

RESOLUTION_MAP = {
    256: {"16:9": (456, 256), "4:3": (336, 256), "1:1": (256, 256), "9:16": (256, 456)},
    480: {"16:9": (848, 480), "4:3": (640, 480), "1:1": (480, 480), "9:16": (480, 848)},
    720: {"16:9": (1280, 720), "4:3": (960, 720), "1:1": (720, 720), "9:16": (720, 1280)},
}


def build_prompt(text, resolution, aspect_ratio):
    w, h = RESOLUTION_MAP[resolution][aspect_ratio]
    return {
        "subjects": [{"description": text}],
        "background_setting": "",
        "lighting": {"conditions": "natural lighting"},
        "aesthetics": {"composition": "well-framed", "color_scheme": "natural colors"},
        "context": text,
        "output_parameters": {
            "height": h,
            "width": w,
            "num_frames": 1,
            "resolution": f"{resolution}p",
            "aspect_ratio": aspect_ratio.replace(":", ","),
        },
    }


def get_input(prompt_text, default=None):
    if default:
        val = input(f"  {prompt_text} [{default}]: ").strip()
        return val if val else default
    return input(f"  {prompt_text}: ").strip()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print()
    print("=" * 60)
    print("  Cosmos3-Nano  |  Text-to-Image Generator  |  GB10")
    print("=" * 60)
    print()

    prompt = get_input("Describe the image")

    resolution = int(get_input("Resolution (256/480/720)", "480"))
    aspect_ratio = get_input("Aspect ratio (16:9 / 4:3 / 1:1 / 9:16)", "16:9")
    steps = int(get_input("Inference steps", "35"))
    guidance = float(get_input("Guidance scale", "6.0"))
    seed = int(get_input("Seed", "42"))
    output_name = get_input("Output filename (leave blank for auto)", "")

    w, h = RESOLUTION_MAP[resolution][aspect_ratio]

    if output_name:
        out_path = output_name if os.path.isabs(output_name) else os.path.join(OUTPUT_DIR, output_name)
        if not out_path.lower().endswith((".jpg", ".png")):
            out_path += ".jpg"
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(OUTPUT_DIR, f"t2i_{timestamp}.jpg")

    print()
    print("-" * 60)
    print(f"  Prompt:     {prompt}")
    print(f"  Resolution: {w}x{h} ({resolution}p {aspect_ratio})")
    print(f"  Steps:      {steps}")
    print(f"  Guidance:   {guidance}")
    print(f"  Seed:       {seed}")
    print(f"  Output:     {out_path}")
    print("-" * 60)
    print()

    json_prompt = build_prompt(prompt, resolution, aspect_ratio)
    negative_prompt = json.load(open(NEGATIVE_PROMPT))

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

    print("  Generating image...")
    t0 = time.time()

    result = pipe(
        prompt=json.dumps(json_prompt),
        negative_prompt=json.dumps(negative_prompt),
        num_frames=1,
        height=h,
        width=w,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    )

    elapsed = time.time() - t0

    image = result.video[0]
    image.save(out_path)

    print()
    print("=" * 60)
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Saved to {out_path}")
    print("=" * 60)

    generate_more = get_input("\n  Generate another with same model? (y/n)", "n")
    while generate_more.lower() in ("y", "yes"):
        print()
        prompt = get_input("  Describe the image")
        seed = int(get_input("  Seed", str(seed + 1)))
        output_name = get_input("  Output filename (leave blank for auto)", "")

        if output_name:
            out_path = output_name if os.path.isabs(output_name) else os.path.join(OUTPUT_DIR, output_name)
            if not out_path.lower().endswith((".jpg", ".png")):
                out_path += ".jpg"
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(OUTPUT_DIR, f"t2i_{timestamp}.jpg")

        json_prompt = build_prompt(prompt, resolution, aspect_ratio)

        print("  Generating image...")
        t0 = time.time()

        result = pipe(
            prompt=json.dumps(json_prompt),
            negative_prompt=json.dumps(negative_prompt),
            num_frames=1,
            height=h,
            width=w,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        )

        elapsed = time.time() - t0
        image = result.video[0]
        image.save(out_path)

        print()
        print(f"  Done in {elapsed:.1f}s")
        print(f"  Saved to {out_path}")

        generate_more = get_input("\n  Generate another? (y/n)", "n")

    print("\n  Goodbye.")


if __name__ == "__main__":
    main()
