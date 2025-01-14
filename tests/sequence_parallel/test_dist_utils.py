import subprocess

import torch
import torch.distributed as dist
import pytest

from turbo_alignment.dist_utils.comm import create_and_broadcast
from turbo_alignment.dist_utils.gather_and_split import gather_and_split

from tests.sequence_parallel.marks import has_two_gpus


INPUTS = [
    [[1, 2], [3]],
    [[1, 2, 3], [4, 5, 6]],
]

OUTPUTS = [
    [[0, 1], [2, 3]],
    [[1, 2, 3], [4, 5, 6]],
]


assert len(INPUTS) == len(OUTPUTS)


def do_test_gather_and_split(test_case):
    dist.init_process_group()
    rank = dist.get_rank()

    input_, output = INPUTS[test_case][rank], OUTPUTS[test_case][rank]

    input_ = torch.tensor(input_, device=f'cuda:{rank}')
    result = gather_and_split(input_, None, pad_value=0, padding_side='left').tolist()
    assert result == output, (result, output)


@pytest.mark.skipif(not has_two_gpus(), reason='at least two gpus are required')
@pytest.mark.parametrize(
    'test_case',
    range(len(INPUTS)),
)
def test_gather_and_split(test_case: int):
    subprocess.check_call(
        ['torchrun', '--nnodes', '1', '--nproc-per-node', '2', __file__, '--mode', 'gather_and_split', '--test-case', str(test_case)]
    )


CREATE_AND_BROADCAST_INPUTS = [
    [1, 2, 3],
    [True, False, True, False],
]


def do_test_create_and_broadcast(test_case: int):
    dist.init_process_group()
    rank = dist.get_rank()
    input_ = CREATE_AND_BROADCAST_INPUTS[test_case]
    device = torch.device(f'cuda:{rank}')
    if rank == 1:
        input_tensor = torch.tensor(input_, device=device)
    else:
        input_tensor = None

    result = create_and_broadcast(
        input_tensor,
        src=1,
        group=None,
        device=device,
    )

    assert result.tolist() == input_


MODE_TO_FUNCTION = {
    'gather_and_split': do_test_gather_and_split,
    'create_and_broadcast': do_test_create_and_broadcast,
}


@pytest.mark.skipif(not has_two_gpus(), reason='at least two gpus are required')
@pytest.mark.parametrize(
    'test_case',
    range(len(CREATE_AND_BROADCAST_INPUTS)),
)
def test_create_and_broadcast(test_case: int):
    subprocess.check_call(
        ['torchrun', '--nnodes', '1', '--nproc-per-node', '2', __file__, '--mode', 'create_and_broadcast', '--test-case', str(test_case)]
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=list(MODE_TO_FUNCTION.keys()))
    parser.add_argument('--test-case', type=int, default=0)
    args = parser.parse_args()
    MODE_TO_FUNCTION[args.mode](args.test_case)
