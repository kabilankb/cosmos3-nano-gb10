"""Apply all GB10/Jetson Thor patches to cosmos-framework.

Run inside the container or on bare metal after installing cosmos-framework:
    python docker/patches/apply_patches.py /path/to/cosmos-framework
"""
import os
import shutil
import sys


def patch_file(filepath, old, new):
    with open(filepath, "r") as f:
        content = f.read()
    if old not in content:
        print(f"  SKIP (already patched or not found): {filepath}")
        return False
    content = content.replace(old, new)
    with open(filepath, "w") as f:
        f.write(content)
    print(f"  PATCHED: {filepath}")
    return True


def main():
    if len(sys.argv) < 2:
        framework_dir = "/workspace/cosmos3/cosmos-framework"
    else:
        framework_dir = sys.argv[1]

    if not os.path.isdir(framework_dir):
        print(f"ERROR: {framework_dir} not found")
        sys.exit(1)

    print(f"Applying patches to: {framework_dir}\n")

    # Patch 1: NVML memory fallback
    print("[1/4] NVML memory query fallback (args.py)")
    patch_file(
        f"{framework_dir}/cosmos_framework/inference/args.py",
        """@cache
def _get_device_memory_bytes() -> int:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()
    return info.total""",
        """@cache
def _get_device_memory_bytes() -> int:
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return info.total
    except pynvml.NVMLError:
        import torch
        return torch.cuda.get_device_properties(0).total_memory""",
    )

    # Patch 2: Video reading (torchvision.io.read_video removed in 0.27)
    print("[2/4] Video reading via PyAV (vision.py)")
    patch_file(
        f"{framework_dir}/cosmos_framework/inference/vision.py",
        """    frames, _, info = torchvision.io.read_video(str(path), pts_unit="sec")
    frames = frames[:max_frames].permute(0, 3, 1, 2).permute(1, 0, 2, 3)
    fps = float(info.get("video_fps", 24.0))""",
        """    import av as _av
    container = _av.open(str(path))
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 24.0
    frame_list = []
    for frame in container.decode(video=0):
        frame_list.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
        if len(frame_list) >= max_frames:
            break
    container.close()
    frames = torch.stack(frame_list).permute(0, 3, 1, 2).permute(1, 0, 2, 3)""",
    )

    # Patch 3: SDPA attention backend
    print("[3/4] SDPA attention backend (frontend.py + backends.py)")
    sdpa_src = os.path.join(os.path.dirname(__file__), "sdpa_fallback.py")
    sdpa_dst = f"{framework_dir}/cosmos_framework/model/attention/sdpa_fallback.py"
    if os.path.exists(sdpa_src):
        shutil.copy2(sdpa_src, sdpa_dst)
        print(f"  COPIED: {sdpa_dst}")
    else:
        print(f"  SKIP: sdpa_fallback.py source not found at {sdpa_src}")

    patch_file(
        f"{framework_dir}/cosmos_framework/model/attention/frontend.py",
        """from cosmos_framework.model.attention.flash2 import flash2_attention
from cosmos_framework.model.attention.flash3 import flash3_attention
from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.model.attention.natten import natten_attention, natten_multi_dim_attention
from cosmos_framework.model.attention.utils.environment import filter_attention_merge_backends
from cosmos_framework.model.attention.utils.safe_ops import log


# Map backend names to their frontend attention API
BACKEND_MAP = {
    "natten": natten_attention,
    "flash2": flash2_attention,
    "flash3": flash3_attention,
}""",
        """from cosmos_framework.model.attention.flash2 import flash2_attention
from cosmos_framework.model.attention.flash3 import flash3_attention
from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.model.attention.natten import natten_attention, natten_multi_dim_attention
from cosmos_framework.model.attention.sdpa_fallback import sdpa_attention
from cosmos_framework.model.attention.utils.environment import filter_attention_merge_backends
from cosmos_framework.model.attention.utils.safe_ops import log


# Map backend names to their frontend attention API
BACKEND_MAP = {
    "natten": natten_attention,
    "flash2": flash2_attention,
    "flash3": flash3_attention,
    "sdpa": sdpa_attention,
}""",
    )

    # Patch backends.py: add SDPA check and backend list entry
    patch_file(
        f"{framework_dir}/cosmos_framework/model/attention/backends.py",
        """BACKEND_CHECK_MAP = {
    "natten": natten_attention_check,
    "flash2": flash2_attention_check,
    "flash3": flash3_attention_check,
}""",
        """def sdpa_attention_check(**kwargs) -> bool:
    return True

BACKEND_CHECK_MAP = {
    "natten": natten_attention_check,
    "flash2": flash2_attention_check,
    "flash3": flash3_attention_check,
    "sdpa": sdpa_attention_check,
}""",
    )

    patch_file(
        f"{framework_dir}/cosmos_framework/model/attention/backends.py",
        """    elif arch_tag >= 80:
        default_backends = [
            "flash2",
            "natten",
        ]
    else:
        default_backends = ["natten"]""",
        """    elif arch_tag >= 80:
        default_backends = [
            "flash2",
            "natten",
            "sdpa",
        ]
    else:
        default_backends = ["natten", "sdpa"]""",
    )

    # Patch 4: Register SO-101 embodiment
    print("[4/4] Register lerobot-so101 embodiment (domain_utils.py)")
    patch_file(
        f"{framework_dir}/cosmos_framework/data/vfm/action/domain_utils.py",
        '    "agibotworld": 15,\n    "fractal": 20,',
        '    "agibotworld": 15,\n    "lerobot-so101": 16,\n    "fractal": 20,',
    )
    patch_file(
        f"{framework_dir}/cosmos_framework/data/vfm/action/domain_utils.py",
        '    "agibotworld": 29,\n    "fractal": 10,',
        '    "agibotworld": 29,\n    "lerobot-so101": 6,\n    "fractal": 10,',
    )

    print("\nAll patches applied.")


if __name__ == "__main__":
    main()
