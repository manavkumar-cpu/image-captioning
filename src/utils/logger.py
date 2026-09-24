"""Console logging. Training file/TensorBoard logging comes in Phase 8."""

import logging
import sys


def get_logger(name: str = "image_captioning") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)

        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

        logger.addHandler(handler)

    return logger