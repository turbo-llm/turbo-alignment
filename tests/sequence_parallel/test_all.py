import os
import contextlib
import subprocess
import tempfile
import sys
import logging

import torch

import tests.sequence_parallel.test_data_loaders  # noqa
import tests.sequence_parallel.test_ulysses  # noqa
import tests.sequence_parallel.test_gemma_model  # noqa
from tests.sequence_parallel.launcher import app

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    app()
