from dataclasses import dataclass
from typing import Sequence

from vllm_guard.training.rl.verl_integration import has_verl


@dataclass(frozen=True)
class VerlLaunchConfig:
    train_files: Sequence[str]
    val_files: Sequence[str]
    output_dir: str
    train_batch_size: int
    rollout_batch_size: int
    extra_args: tuple[str, ...] = ()


def build_verl_command(config: VerlLaunchConfig) -> list[str]:
    if not has_verl():
        raise FileNotFoundError(
            "verl is required for RL launch. Install it in the active environment "
            "or vendor it under third_party/verl."
        )
    cmd = [
        "python",
        "-m",
        "verl.trainer.main_ppo",
        f"data.train_files={','.join(config.train_files)}",
        f"data.val_files={','.join(config.val_files)}",
        f"trainer.default_hdfs_dir={config.output_dir}",
        f"data.train_batch_size={config.train_batch_size}",
        f"data.rollout_batch_size={config.rollout_batch_size}",
        "algorithm.adv_estimator=grpo",
    ]
    cmd.extend(config.extra_args)
    return cmd
