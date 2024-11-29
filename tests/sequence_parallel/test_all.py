import os
import contextlib
import subprocess
import tempfile
import sys
import logging

import pytest
import torch

import tests.sequence_parallel.test_data_loaders  # noqa
import tests.sequence_parallel.test_ulysses  # noqa
import tests.sequence_parallel.test_gemma_model  # noqa
from tests.sequence_parallel.launcher import app

logger = logging.getLogger(__name__)


def launch_with_name(name: str, num_gpus: int):
    args = [
        'deepspeed',
        '--no_local_rank',
    ]

    with contextlib.ExitStack() as stack:
        if os.getenv('CUDA_VISIBLE_DEVICES'):
            logger.info('CUDA_VISIBLE_DEVICES is set, so do not pass --num_gpus argument, but create hostfile')
            temp_dir = stack.enter_context(tempfile.TemporaryDirectory())
            host_file = os.path.join(temp_dir, 'hostfile')
            with open(host_file, 'w') as output:
                output.write(f'localhost slots={torch.cuda.device_count()}\n')

            args.extend(['--hostfile', host_file, '--no_ssh_check', '--master_addr', 'localhost'])

        else:
            args.extend(['--num_gpus', str(num_gpus)])

        args.extend([__file__, name,])

        subprocess.check_call(
            args,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )


@pytest.mark.parametrize(
    'name',
    [c.name for c in app.registered_commands]
)
def test_deepspeed_scripts(name):
    launch_with_name(name, 2)


if __name__ == '__main__':
    app()
