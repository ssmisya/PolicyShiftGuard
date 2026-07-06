import importlib.util
from pathlib import Path

from vllm_guard.common.constants import REPO_ROOT


def get_vendored_verl_root() -> Path:
    return REPO_ROOT / "third_party" / "verl"


def get_vendored_verl_python_package() -> Path:
    return get_vendored_verl_root() / "verl"


def has_vendored_verl() -> bool:
    root = get_vendored_verl_root()
    return root.exists() and (root / "setup.py").exists() and get_vendored_verl_python_package().exists()


def has_installed_verl() -> bool:
    return importlib.util.find_spec("verl") is not None


def has_verl() -> bool:
    return has_installed_verl() or has_vendored_verl()


def describe_verl_source() -> str:
    if has_installed_verl():
        return "installed Python package 'verl'"
    if has_vendored_verl():
        return str(get_vendored_verl_root())
    return "not found"
