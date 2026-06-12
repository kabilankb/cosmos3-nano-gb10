"""Cosmos3-Nano text-to-video test on GB10."""
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video
import os

BASE = os.environ.get("COSMOS3_DIR", "/workspace/cosmos3")
ASSETS = f"{BASE}/Cosmos3-Nano-assets/assets"
MODEL = f"{BASE}/Cosmos3-Nano"
OUTPUT = f"{BASE}/output"
os.makedirs(OUTPUT, exist_ok=True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

json_prompt = json.load(open(f"{ASSETS}/example_t2v_prompt.json"))
negative_prompt = json.load(open(f"{ASSETS}/negative_prompt.json"))

print("Loading Cosmos3-Nano pipeline...")
pipe = Cosmos3OmniPipeline.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    enable_safety_checker=False,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0
)
print("Pipeline loaded. Generating video...")

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    num_frames=57,
    height=480,
    width=848,
    num_inference_steps=35,
    guidance_scale=6.0,
    generator=torch.Generator(device="cuda").manual_seed(42),
)

out_path = f"{OUTPUT}/cosmos3_nano_t2v_test.mp4"
export_to_video(result.video, out_path, fps=24)
print(f"Saved to {out_path}")
