from __future__ import annotations
import logging
import sys
from dataclasses import dataclass


@dataclass
class LogConfig:
    level: int = logging.INFO
    name: str = "torch_speedkit"


def setup_logger(cfg: LogConfig = LogConfig()) -> logging.Logger:
    logger = logging.getLogger(cfg.name)
    logger.setLevel(cfg.level)
    logger.propagate = False

    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(cfg.level)
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        h.setFormatter(fmt)
        logger.addHandler(h)

    return logger
