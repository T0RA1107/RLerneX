import sys

from loguru import logger
from omegaconf import DictConfig


def init_logger(cfg: DictConfig):
    logging_cfg = cfg.logger
    logger.remove()
    logger.add(
        sys.stdout,
        backtrace=True,
        format=logging_cfg.formatters.loguru.format,
        level=logging_cfg.root.level,
    )
    logger.add(
        logging_cfg.filename,
        backtrace=True,
        format=logging_cfg.formatters.loguru.format,
        level=logging_cfg.root.level,
    )
