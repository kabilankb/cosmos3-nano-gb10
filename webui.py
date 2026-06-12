"""Cosmos3-Nano Web UI — Text/Image/Video to Video/Image generation."""
import json
import os
import shutil
import subprocess
import threading
import time
import traceback

import gradio as gr
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image

BASE = os.environ.get("COSMOS3_DIR", os.path.expanduser("~/cosmos3"))
MODEL = os.environ.get("COSMOS3_MODEL", f"{BASE}/Cosmos3-Nano")
ASSETS = f"{BASE}/Cosmos3-Nano/assets"
if not os.path.exists(ASSETS):
    ASSETS = f"{BASE}/Cosmos3-Nano-assets/assets"
OUTPUT = f"{BASE}/output"
os.makedirs(OUTPUT, exist_ok=True)

NEGATIVE_PROMPT_PATH = f"{ASSETS}/negative_prompt.json"

RESOLUTIONS = {
    "256p 16:9 (456x256)": (456, 256),
    "256p 1:1 (256x256)": (256, 256),
    "256p 9:16 (256x456)": (256, 456),
    "480p 16:9 (848x480)": (848, 480),
    "480p 4:3 (640x480)": (640, 480),
    "480p 1:1 (480x480)": (480, 480),
    "480p 9:16 (480x848)": (480, 848),
    "720p 16:9 (1280x720)": (1280, 720),
    "720p 4:3 (960x720)": (960, 720),
    "720p 1:1 (720x720)": (720, 720),
    "720p 9:16 (720x1280)": (720, 1280),
}

pipe = None
pipe_lock = threading.Lock()


def load_pipeline():
    global pipe
    if pipe is not None:
        return pipe
    print("Loading Cosmos3-Nano pipeline...", flush=True)
    pipe = Cosmos3OmniPipeline.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        enable_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=10.0
    )
    print("Pipeline loaded.", flush=True)
    return pipe


def reencode_for_browser(src, dst, fps=24):
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        "-r", str(fps),
        dst,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print(f"[ffmpeg] FAILED (rc={r.returncode}): {r.stderr}", flush=True)
        shutil.copy2(src, dst)
    else:
        print(f"[ffmpeg] OK: {dst}", flush=True)


def build_prompt(text, mode, width, height, num_frames, fps):
    from math import gcd
    res_val = 480
    for r in [256, 480, 720]:
        if height <= r:
            res_val = r
            break

    g = gcd(width, height)
    ar_w, ar_h = width // g, height // g
    duration = num_frames / fps if fps > 0 else 1
    time_range = f"0:00-0:{duration:04.1f}"

    prompt = {
        "subjects": [{"description": text, "action": text}],
        "background_setting": "",
        "lighting": {"conditions": "natural", "direction": "ambient"},
        "aesthetics": {"composition": "balanced", "color_scheme": "natural"},
        "context": text,
        "actions": [{"time": time_range, "description": text}],
        "segments": [{"segment_index": 0, "time_range": time_range, "description": text}],
        "temporal_caption": text,
        "output_parameters": {
            "height": height, "width": width,
            "num_frames": num_frames, "fps": fps,
            "resolution": f"{res_val}p",
            "aspect_ratio": f"{ar_w},{ar_h}",
        },
    }
    if mode == "t2i":
        prompt["output_parameters"]["num_frames"] = 1
        prompt["output_parameters"].pop("fps", None)
    if mode != "t2i":
        prompt["cinematography"] = {
            "camera_motion": "static",
            "framing": "medium shot",
            "camera_angle": "eye level",
        }
    return prompt


def load_negative_prompt():
    if os.path.exists(NEGATIVE_PROMPT_PATH):
        return json.load(open(NEGATIVE_PROMPT_PATH))
    return {"subjects": [{"description": "blurry, low quality, distorted"}]}


def _run_pipeline(p, **kwargs):
    """Run pipeline in a thread so the main thread can yield keepalives."""
    result = [None]
    error = [None]

    def _worker():
        try:
            result[0] = p(**kwargs)
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_worker)
    t.start()
    return t, result, error


def generate_t2v(text, resolution, num_frames, fps, steps, guidance, seed):
    if not text or not text.strip():
        raise gr.Error("Please enter a prompt.")
    try:
        with pipe_lock:
            p = load_pipeline()
            w, h = RESOLUTIONS[resolution]
            num_frames, fps, steps, seed = int(num_frames), int(fps), int(steps), int(seed)

            json_prompt = build_prompt(text, "t2v", w, h, num_frames, fps)
            neg = load_negative_prompt()
            gen = torch.Generator(device="cuda").manual_seed(seed)

            print(f"[T2V] {w}x{h}, {num_frames}f, {steps}s, seed={seed}", flush=True)
            t, result, error = _run_pipeline(
                p, prompt=json.dumps(json_prompt), negative_prompt=json.dumps(neg),
                num_frames=num_frames, height=h, width=w,
                num_inference_steps=steps, guidance_scale=float(guidance), generator=gen,
            )

            elapsed = 0
            while t.is_alive():
                t.join(timeout=3)
                elapsed += 3
                yield gr.skip(), gr.skip(), f"Generating... {elapsed}s"

            if error[0]:
                raise error[0]

            ts = int(time.time())
            raw = os.path.join(OUTPUT, f"cosmos3_t2v_raw_{ts}.mp4")
            out = os.path.join(OUTPUT, f"cosmos3_t2v_{ts}_{seed}.mp4")
            export_to_video(result[0].video, raw, fps=fps)
            reencode_for_browser(raw, out, fps=fps)
            if os.path.exists(raw) and raw != out:
                os.remove(raw)

            sz = os.path.getsize(out)
            print(f"[T2V] Done: {out} ({sz} bytes)", flush=True)
            yield out, out, f"Done! {sz//1024}KB"
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Failed: {e}")


def generate_i2v(image, text, resolution, num_frames, fps, steps, guidance, seed):
    if image is None:
        raise gr.Error("Please upload an image.")
    try:
        with pipe_lock:
            p = load_pipeline()
            w, h = RESOLUTIONS[resolution]
            num_frames, fps, steps, seed = int(num_frames), int(fps), int(steps), int(seed)

            prompt_text = text.strip() if text else "Animate this image"
            json_prompt = build_prompt(prompt_text, "i2v", w, h, num_frames, fps)
            neg = load_negative_prompt()
            gen = torch.Generator(device="cuda").manual_seed(seed)
            pil_image = load_image(image)

            print(f"[I2V] {w}x{h}, {num_frames}f, {steps}s, seed={seed}", flush=True)
            t, result, error = _run_pipeline(
                p, prompt=json.dumps(json_prompt), negative_prompt=json.dumps(neg),
                image=pil_image, num_frames=num_frames, height=h, width=w,
                num_inference_steps=steps, guidance_scale=float(guidance), generator=gen,
            )

            elapsed = 0
            while t.is_alive():
                t.join(timeout=3)
                elapsed += 3
                yield gr.skip(), gr.skip(), f"Generating... {elapsed}s"

            if error[0]:
                raise error[0]

            ts = int(time.time())
            raw = os.path.join(OUTPUT, f"cosmos3_i2v_raw_{ts}.mp4")
            out = os.path.join(OUTPUT, f"cosmos3_i2v_{ts}_{seed}.mp4")
            export_to_video(result[0].video, raw, fps=fps)
            reencode_for_browser(raw, out, fps=fps)
            if os.path.exists(raw) and raw != out:
                os.remove(raw)

            sz = os.path.getsize(out)
            print(f"[I2V] Done: {out} ({sz} bytes)", flush=True)
            yield out, out, f"Done! {sz//1024}KB"
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Failed: {e}")


def generate_t2i(text, resolution, steps, guidance, seed):
    if not text or not text.strip():
        raise gr.Error("Please enter a prompt.")
    try:
        with pipe_lock:
            p = load_pipeline()
            w, h = RESOLUTIONS[resolution]
            steps, seed = int(steps), int(seed)

            json_prompt = build_prompt(text, "t2i", w, h, 1, 1)
            neg = load_negative_prompt()
            gen = torch.Generator(device="cuda").manual_seed(seed)

            print(f"[T2I] {w}x{h}, {steps}s, seed={seed}", flush=True)
            t, result, error = _run_pipeline(
                p, prompt=json.dumps(json_prompt), negative_prompt=json.dumps(neg),
                num_frames=1, height=h, width=w,
                num_inference_steps=steps, guidance_scale=float(guidance), generator=gen,
            )

            elapsed = 0
            while t.is_alive():
                t.join(timeout=3)
                elapsed += 3
                yield gr.skip(), gr.skip(), f"Generating... {elapsed}s"

            if error[0]:
                raise error[0]

            img = result[0].video[0]
            ts = int(time.time())
            out = os.path.join(OUTPUT, f"cosmos3_t2i_{ts}_{seed}.jpg")
            img.save(out)

            sz = os.path.getsize(out)
            print(f"[T2I] Done: {out} ({sz} bytes)", flush=True)
            yield out, out, f"Done! {sz//1024}KB"
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Failed: {e}")


NVIDIA_CSS = """
:root {
    --nv-green: #76B900;
    --nv-green-dark: #5a8f00;
    --nv-green-light: #8ed600;
    --nv-black: #1a1a1a;
    --nv-dark: #222222;
    --nv-gray: #2d2d2d;
    --nv-lgray: #3a3a3a;
    --nv-text: #e0e0e0;
}
.gradio-container {
    background: linear-gradient(135deg, #0d0d0d 0%, #1a1a1a 50%, #0d1a00 100%) !important;
    max-width: 1400px !important;
}
#nvidia-header {
    background: linear-gradient(90deg, var(--nv-green) 0%, var(--nv-green-dark) 100%);
    padding: 20px 30px; border-radius: 12px; margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(118,185,0,0.3);
}
#nvidia-header h1 { color: white !important; margin: 0 !important; font-size: 28px !important; font-weight: 700 !important; }
#nvidia-header p { color: rgba(255,255,255,0.9) !important; margin: 4px 0 0 0 !important; font-size: 14px !important; }
.tab-nav button {
    background: var(--nv-gray) !important; color: var(--nv-text) !important;
    border: 1px solid var(--nv-lgray) !important; border-radius: 8px 8px 0 0 !important;
    font-weight: 600 !important; padding: 10px 24px !important;
}
.tab-nav button.selected {
    background: var(--nv-green) !important; color: white !important;
    border-color: var(--nv-green) !important; box-shadow: 0 2px 10px rgba(118,185,0,0.4) !important;
}
.tab-nav button:hover:not(.selected) {
    background: var(--nv-lgray) !important; border-color: var(--nv-green) !important;
}
.primary {
    background: linear-gradient(135deg, var(--nv-green) 0%, var(--nv-green-dark) 100%) !important;
    border: none !important; color: white !important; font-weight: 700 !important;
    font-size: 16px !important; padding: 12px 24px !important; border-radius: 8px !important;
    box-shadow: 0 4px 15px rgba(118,185,0,0.3) !important; text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
.primary:hover {
    background: linear-gradient(135deg, var(--nv-green-light) 0%, var(--nv-green) 100%) !important;
    box-shadow: 0 6px 25px rgba(118,185,0,0.5) !important; transform: translateY(-1px) !important;
}
textarea, input[type="text"], input[type="number"] {
    background: var(--nv-gray) !important; border: 1px solid var(--nv-lgray) !important;
    color: var(--nv-text) !important; border-radius: 6px !important;
}
textarea:focus, input:focus {
    border-color: var(--nv-green) !important; box-shadow: 0 0 0 2px rgba(118,185,0,0.2) !important;
}
input[type="range"] { accent-color: var(--nv-green) !important; }
.upload-area, .drop-area {
    border: 2px dashed var(--nv-green) !important; background: rgba(118,185,0,0.05) !important;
    border-radius: 8px !important;
}
#nvidia-footer {
    text-align: center; padding: 12px; color: #888; font-size: 12px;
    border-top: 1px solid var(--nv-lgray); margin-top: 16px;
}
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: var(--nv-black); }
::-webkit-scrollbar-thumb { background: var(--nv-green-dark); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--nv-green); }
"""

nvidia_theme = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#f0f9e0", c100="#d4f0a0", c200="#b8e760", c300="#9cde20",
        c400="#8ed600", c500="#76B900", c600="#5a8f00", c700="#3e6500",
        c800="#223b00", c900="#111e00", c950="#0a1200",
    ),
    neutral_hue=gr.themes.Color(
        c50="#f5f5f5", c100="#e0e0e0", c200="#b0b0b0", c300="#808080",
        c400="#606060", c500="#3a3a3a", c600="#2d2d2d", c700="#222222",
        c800="#1a1a1a", c900="#111111", c950="#0d0d0d",
    ),
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="*neutral_950",
    body_background_fill_dark="*neutral_950",
    block_background_fill="*neutral_800",
    block_background_fill_dark="*neutral_800",
    block_border_color="*neutral_600",
    block_label_text_color="*neutral_100",
    block_title_text_color="*neutral_50",
    input_background_fill="*neutral_700",
    input_background_fill_dark="*neutral_700",
    input_border_color="*neutral_600",
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_400",
    button_primary_text_color="white",
)

NVIDIA_LOGO_SVG = """<svg width="40" height="30" viewBox="0 0 351 264" fill="white" xmlns="http://www.w3.org/2000/svg">
<path d="M121.7 45.1v-19.3c1.5-.2 3-.3 4.6-.3 44.6-2 77.4 36.4 77.4 36.4s-41.6 47-68.2 47c-5.2 0-9.8-1.2-13.8-3.3V63.2c20.6-2.6 24.8 12.8 47.2 37.8 0 0 16.2-13.6 27.4-28.2 0 0-22.2-28.8-51.2-28.8-8 0-15.6 1.2-23.4 1.1zm0-37.3V26c73.2-5.8 118.4 47 118.4 47S188.7 135.6 145 135.6c-8.2 0-15.8-1.6-23.4-4.2v16.8c6.6 1.2 13.4 1.8 20.4 1.8 55.6 0 96.4-28.4 135.8-61.6 6.5 5.3 33.4 18 38.8 23.4-37 30.2-123.6 65.4-177.4 65.4-5.8 0-11.6-.4-17.6-1V7.8h.1zM0 177.3h351V264H0v-86.7z"/>
</svg>"""

with gr.Blocks(title="Cosmos3-Nano | NVIDIA") as app:
    with gr.Column(elem_id="nvidia-header"):
        gr.HTML(f"""
            <div style="display:flex; align-items:center; gap:16px;">
                {NVIDIA_LOGO_SVG}
                <div>
                    <h1>Cosmos3-Nano Generator</h1>
                    <p>16B Omnimodal World Foundation Model &mdash; Text-to-Video | Image-to-Video | Text-to-Image</p>
                </div>
            </div>
        """)

    with gr.Tabs():
        with gr.Tab("Text to Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    t2v_text = gr.Textbox(label="Prompt", placeholder="A robot arm picking up a red cube...", lines=4)
                    t2v_res = gr.Dropdown(choices=list(RESOLUTIONS.keys()), value="480p 16:9 (848x480)", label="Resolution")
                    with gr.Row():
                        t2v_frames = gr.Slider(25, 1025, value=121, step=1, label="Frames")
                        t2v_fps = gr.Slider(8, 30, value=24, step=1, label="FPS")
                    with gr.Row():
                        t2v_steps = gr.Slider(10, 50, value=35, step=1, label="Steps")
                        t2v_guidance = gr.Slider(1.0, 15.0, value=6.0, step=0.5, label="Guidance")
                    t2v_seed = gr.Number(value=42, label="Seed", precision=0)
                    t2v_btn = gr.Button("Generate Video", variant="primary")
                    t2v_status = gr.Textbox(label="Status", interactive=False)
                with gr.Column(scale=1):
                    t2v_video = gr.Video(label="Preview", autoplay=True)
                    t2v_file = gr.File(label="Download")

            t2v_btn.click(
                generate_t2v,
                inputs=[t2v_text, t2v_res, t2v_frames, t2v_fps, t2v_steps, t2v_guidance, t2v_seed],
                outputs=[t2v_video, t2v_file, t2v_status],
            )

        with gr.Tab("Image to Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    i2v_image = gr.Image(label="Drop Image Here", type="filepath")
                    i2v_text = gr.Textbox(label="Prompt (optional)", placeholder="Describe the motion...", lines=3)
                    i2v_res = gr.Dropdown(choices=list(RESOLUTIONS.keys()), value="480p 16:9 (848x480)", label="Resolution")
                    with gr.Row():
                        i2v_frames = gr.Slider(25, 1025, value=121, step=1, label="Frames")
                        i2v_fps = gr.Slider(8, 30, value=24, step=1, label="FPS")
                    with gr.Row():
                        i2v_steps = gr.Slider(10, 50, value=35, step=1, label="Steps")
                        i2v_guidance = gr.Slider(1.0, 15.0, value=6.0, step=0.5, label="Guidance")
                    i2v_seed = gr.Number(value=42, label="Seed", precision=0)
                    i2v_btn = gr.Button("Generate Video", variant="primary")
                    i2v_status = gr.Textbox(label="Status", interactive=False)
                with gr.Column(scale=1):
                    i2v_video = gr.Video(label="Preview", autoplay=True)
                    i2v_file = gr.File(label="Download")

            i2v_btn.click(
                generate_i2v,
                inputs=[i2v_image, i2v_text, i2v_res, i2v_frames, i2v_fps, i2v_steps, i2v_guidance, i2v_seed],
                outputs=[i2v_video, i2v_file, i2v_status],
            )

        with gr.Tab("Text to Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    t2i_text = gr.Textbox(label="Prompt", placeholder="A futuristic city skyline at sunset...", lines=4)
                    t2i_res = gr.Dropdown(choices=list(RESOLUTIONS.keys()), value="480p 16:9 (848x480)", label="Resolution")
                    with gr.Row():
                        t2i_steps = gr.Slider(10, 50, value=35, step=1, label="Steps")
                        t2i_guidance = gr.Slider(1.0, 15.0, value=6.0, step=0.5, label="Guidance")
                    t2i_seed = gr.Number(value=42, label="Seed", precision=0)
                    t2i_btn = gr.Button("Generate Image", variant="primary")
                    t2i_status = gr.Textbox(label="Status", interactive=False)
                with gr.Column(scale=1):
                    t2i_output = gr.Image(label="Preview")
                    t2i_file = gr.File(label="Download")

            t2i_btn.click(
                generate_t2i,
                inputs=[t2i_text, t2i_res, t2i_steps, t2i_guidance, t2i_seed],
                outputs=[t2i_output, t2i_file, t2i_status],
            )

    gr.HTML(f"""
        <div id="nvidia-footer">
            NVIDIA Cosmos3-Nano &bull; Model: <code>{MODEL}</code> &bull; Output: <code>{OUTPUT}</code>
        </div>
    """)

if __name__ == "__main__":
    host = os.environ.get("WEBUI_HOST", "0.0.0.0")
    port = int(os.environ.get("WEBUI_PORT", "7860"))
    print(f"Cosmos3-Nano Web UI: http://0.0.0.0:{port}", flush=True)
    print(f"Model: {MODEL}", flush=True)
    print(f"Output: {OUTPUT}", flush=True)
    app.queue(default_concurrency_limit=1)
    app.launch(
        server_name=host,
        server_port=port,
        share=False,
        allowed_paths=["/tmp", OUTPUT],
        theme=nvidia_theme,
        css=NVIDIA_CSS,
    )
