from pathlib import Path
import os


MODEL_REGISTRY = {
    "llava-1.5-7b": {
        "name": "LLaVA-1.5-7B",
        "paths": ["liuhaotian/llava-v1.5-7b"],
        "type": "vllm",
    },
    "llava-1.5-13b": {
        "name": "LLaVA-1.5-13B",
        "paths": ["liuhaotian/llava-v1.5-13b"],
        "type": "vllm",
    },
    "llava-1.6-7b": {
        "name": "LLaVA-1.6-Vicuna-7B",
        "paths": ["liuhaotian/llava-v1.6-vicuna-7b"],
        "type": "vllm",
    },
    "llava-1.6-13b": {
        "name": "LLaVA-1.6-Vicuna-13B",
        "paths": ["liuhaotian/llava-v1.6-vicuna-13b"],
        "type": "vllm",
    },
    "llava-onevision-qwen2-7b": {
        "name": "LLaVA-OneVision-Qwen2-7B",
        "paths": ["lmms-lab/llava-onevision-qwen2-7b-ov"],
        "type": "vllm",
    },
    "qwen-vl-chat": {
        "name": "Qwen-VL-Chat",
        "paths": ["Qwen/Qwen-VL-Chat"],
        "type": "vllm",
    },
    "qwen2-vl-2b": {
        "name": "Qwen2-VL-2B-Instruct",
        "paths": ["Qwen/Qwen2-VL-2B-Instruct"],
        "type": "vllm",
    },
    "qwen2-vl-7b": {
        "name": "Qwen2-VL-7B-Instruct",
        "paths": ["Qwen/Qwen2-VL-7B-Instruct"],
        "type": "vllm",
    },
    "qwen2-vl-72b": {
        "name": "Qwen2-VL-72B-Instruct",
        "paths": ["Qwen/Qwen2-VL-72B-Instruct"],
        "type": "vllm",
    },
    "qwen2.5-vl-3b": {
        "name": "Qwen2.5-VL-3B-Instruct",
        "paths": ["Qwen/Qwen2.5-VL-3B-Instruct"],
        "type": "vllm",
    },
    "qwen2.5-vl-7b": {
        "name": "Qwen2.5-VL-7B-Instruct",
        "paths": ["Qwen/Qwen2.5-VL-7B-Instruct"],
        "type": "vllm",
    },
    "qwen2.5-vl-32b": {
        "name": "Qwen2.5-VL-32B-Instruct",
        "paths": ["Qwen/Qwen2.5-VL-32B-Instruct"],
        "type": "vllm",
    },
    "qwen2.5-vl-72b": {
        "name": "Qwen2.5-VL-72B-Instruct",
        "paths": ["Qwen/Qwen2.5-VL-72B-Instruct"],
        "type": "vllm",
    },
    "llava-guard-7b": {
        "name": "LLaVA-Guard-7B",
        "paths": ["liuhaotian/llava-guard-7b"],
        "type": "llavaguard",
    },
    "llava-guard-13b": {
        "name": "LLaVA-Guard-13B",
        "paths": ["liuhaotian/llava-guard-13b"],
        "type": "llavaguard",
    },
}


def get_model_info(model_name: str) -> dict:
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]
    if os.path.exists(model_name):
        return {"name": Path(model_name).name, "paths": [model_name], "type": "vllm"}
    return {"name": model_name, "paths": [model_name], "type": "vllm"}


def resolve_model_path(model_name: str) -> tuple[str, str, dict]:
    info = get_model_info(model_name)
    for path in info["paths"]:
        if path.startswith("/") and os.path.exists(path):
            return path, "local", info
    return info["paths"][0], "hub", info
