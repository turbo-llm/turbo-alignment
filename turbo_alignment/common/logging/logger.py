from __future__ import annotations

import logging
import os
import sys

import loguru

config = {
    'handlers': [
        {
            'sink': sys.stdout,
            'serialize': False,
            'colorize': True,
            'format': '{file}:{line} [<level>{level}</level>] {time:YYYY-MM-DDTHH:mm:ssZ} {message}',
            'level': os.getenv('LOGGING_LEVEL', 'INFO'),
        }
    ]
}


def get_project_logger() -> loguru.Logger:
    from loguru import logger

    logging.basicConfig(level=logging.INFO)

    logging.getLogger('alignment-logger')

    logger.configure(**config)  # type: ignore[arg-type]

    return logger
