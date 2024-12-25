import contextlib
import os
import sys
import subprocess
import logging
import tempfile

import torch
import typer

logger = logging.getLogger(__name__)

app = typer.Typer(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    pretty_exceptions_enable=False,
)


def launch_with_name(name: str, num_gpus: int, env: dict[str, str] | None = None):
    args = [
        'deepspeed',
        '--no_local_rank',
    ]

    with contextlib.ExitStack() as stack:
        if os.getenv('CUDA_VISIBLE_DEVICES'):
            logger.info('CUDA_VISIBLE_DEVICES is set, so do not pass --num_gpus argument, but create hostfile')
            temp_dir = stack.enter_context(tempfile.TemporaryDirectory())
            host_file = os.path.join(temp_dir, 'hostfile')
            with open(host_file, 'w', encoding='utf-8') as output:
                output.write(f'localhost slots={torch.cuda.device_count()}\n')

            args.extend(['--hostfile', host_file, '--no_ssh_check', '--master_addr', 'localhost'])

        else:
            args.extend(['--num_gpus', str(num_gpus)])

        args.extend([__file__, name])

        subprocess.check_call(
            args,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=env,
        )
