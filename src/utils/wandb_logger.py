"""WandB logging utilities for experiment tracking."""
from typing import Any, Optional

import wandb
from omegaconf import DictConfig, OmegaConf


def init_wandb(cfg: DictConfig) -> Optional[wandb.Run]:
    """Initialize Weights & Biases logging.

    Args:
        cfg: Configuration object containing wandb settings

    Returns:
        WandB run object if enabled, None otherwise
    """
    if not cfg.wandb.enabled:
        return None

    # Convert OmegaConf to dictionary for wandb
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # Initialize wandb
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name,
        tags=list(cfg.wandb.tags) if cfg.wandb.tags else None,
        config=config_dict,
    )

    return run


def log_metrics(metrics: dict[str, Any], step: int, enabled: bool = True) -> None:
    """Log metrics to WandB.

    Args:
        metrics: Dictionary of metrics to log
        step: Current step/episode number
        enabled: Whether logging is enabled
    """
    if enabled and wandb.run is not None:
        wandb.log(metrics, step=step)


def finish_wandb(enabled: bool = True) -> None:
    """Finish WandB run.

    Args:
        enabled: Whether logging is enabled
    """
    if enabled and wandb.run is not None:
        wandb.finish()
