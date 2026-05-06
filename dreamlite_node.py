import os
import sys
import gc
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image


_PIPELINE_CACHE = {}


def _module_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _candidate_repo_paths(user_repo_path: str):
    paths = []
    if user_repo_path:
        paths.append(user_repo_path)

    base = _module_dir()
    paths.extend([
        os.path.join(base, "DreamLite"),
        os.path.join(base, "third_party", "DreamLite"),
        os.path.join(os.path.dirname(base), "DreamLite"),
    ])
    return paths


def _add_dreamlite_repo_to_syspath(user_repo_path: str) -> Optional[str]:
    for path in _candidate_repo_paths(user_repo_path):
        abs_path = os.path.abspath(os.path.expanduser(path))
        if os.path.isdir(abs_path) and os.path.isdir(os.path.join(abs_path, "dreamlite")):
            if abs_path not in sys.path:
                sys.path.insert(0, abs_path)
            return abs_path
    return None


def _import_pipeline(variant: str, dreamlite_repo: str):
    repo = _add_dreamlite_repo_to_syspath(dreamlite_repo)
    try:
        if variant == "mobile":
            from dreamlite import DreamLiteMobilePipeline
            return DreamLiteMobilePipeline, repo
        from dreamlite import DreamLitePipeline
        return DreamLitePipeline, repo
    except Exception as exc:
        raise RuntimeError(
            "Could not import DreamLite. Clone ByteVisionLab/DreamLite and set "
            "dreamlite_repo to that folder, or place the DreamLite repo inside this "
            "custom node folder. Original error: %r" % (exc,)
        )


def _torch_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device is cuda, but CUDA is not available in this environment.")
    return device


def _resolve_model_path(model_path: str, dreamlite_repo: str) -> str:
    expanded = os.path.expanduser(model_path)
    candidates = [expanded, os.path.abspath(expanded)]

    for repo in _candidate_repo_paths(dreamlite_repo):
        abs_repo = os.path.abspath(os.path.expanduser(repo))
        candidates.append(os.path.join(abs_repo, model_path))

    base = _module_dir()
    candidates.extend([
        os.path.join(base, model_path),
        os.path.join(os.path.dirname(base), model_path),
    ])

    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    # Let from_pretrained raise the final, detailed error.
    return model_path


def _cache_key(model_path: str, repo_path: str, variant: str, dtype_name: str, device: str) -> Tuple[str, str, str, str, str]:
    return (
        os.path.abspath(os.path.expanduser(model_path)),
        os.path.abspath(os.path.expanduser(repo_path)) if repo_path else "",
        variant,
        dtype_name,
        device,
    )


def _load_pipeline(model_path: str, dreamlite_repo: str, variant: str, dtype_name: str, device: str):
    resolved_model_path = _resolve_model_path(model_path, dreamlite_repo)
    key = _cache_key(resolved_model_path, dreamlite_repo, variant, dtype_name, device)

    if key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[key]

    pipeline_cls, resolved_repo = _import_pipeline(variant, dreamlite_repo)
    torch_dtype = _torch_dtype(dtype_name)

    pipe = pipeline_cls.from_pretrained(resolved_model_path, torch_dtype=torch_dtype)
    pipe = pipe.to(device)

    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    _PIPELINE_CACHE[key] = pipe
    return pipe


def _comfy_image_to_pil(image) -> Optional[Image.Image]:
    if image is None:
        return None

    tensor = image
    if isinstance(tensor, (list, tuple)):
        if not tensor:
            return None
        tensor = tensor[0]

    if not torch.is_tensor(tensor):
        raise TypeError("Expected ComfyUI IMAGE tensor for image input.")

    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Expected image tensor with shape [H,W,C] or [B,H,W,C], got {tuple(tensor.shape)}")

    arr = tensor.detach().cpu().float().clamp(0.0, 1.0).numpy()
    arr = (arr * 255.0).round().astype(np.uint8)

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]

    return Image.fromarray(arr, mode="RGB")


def _pil_to_comfy_image(image: Image.Image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None,]


class DreamLiteGenerateEdit:
    """ComfyUI wrapper around ByteVisionLab/DreamLite diffusers-style pipelines."""

    CATEGORY = "DreamLite"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dreamlite_repo": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to the cloned ByteVisionLab/DreamLite repo. Leave blank if dreamlite is already importable or repo is nested in this custom node folder.",
                }),
                "model_path": ("STRING", {
                    "default": "models/DreamLite-base",
                    "multiline": False,
                    "tooltip": "Local path to DreamLite-base or DreamLite-mobile weights.",
                }),
                "variant": (["base", "mobile"], {"default": "base"}),
                "prompt": ("STRING", {"default": "a dog running on the grass", "multiline": True}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "size_mode": (["use_widget_size", "use_input_image_size"], {"default": "use_widget_size"}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": "0 uses the official default: base=28, mobile=4."}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "image_guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 18446744073709551615}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    def generate(
        self,
        dreamlite_repo: str,
        model_path: str,
        variant: str,
        prompt: str,
        width: int,
        height: int,
        size_mode: str,
        steps: int,
        guidance_scale: float,
        image_guidance_scale: float,
        seed: int,
        device: str,
        dtype: str,
        image=None,
    ):
        device = _resolve_device(device)
        steps = int(steps) if int(steps) > 0 else (4 if variant == "mobile" else 28)

        input_pil = _comfy_image_to_pil(image)
        if input_pil is not None and size_mode == "use_input_image_size":
            width, height = input_pil.size

        pipe = _load_pipeline(model_path, dreamlite_repo, variant, dtype, device)
        generator = torch.Generator("cpu").manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)

        kwargs = {
            "prompt": prompt,
            "image": input_pil,
            "height": int(height),
            "width": int(width),
            "num_inference_steps": steps,
            "generator": generator,
        }

        if variant == "base":
            kwargs["guidance_scale"] = float(guidance_scale)
            kwargs["image_guidance_scale"] = float(image_guidance_scale)

        with torch.inference_mode():
            result = pipe(**kwargs)

        out = result.images[0]
        if out.size != (int(width), int(height)):
            out = out.resize((int(width), int(height)), Image.Resampling.LANCZOS)

        return (_pil_to_comfy_image(out),)


class DreamLiteClearCache:
    CATEGORY = "DreamLite"
    RETURN_TYPES = ()
    FUNCTION = "clear"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def clear(self):
        _PIPELINE_CACHE.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return ()
