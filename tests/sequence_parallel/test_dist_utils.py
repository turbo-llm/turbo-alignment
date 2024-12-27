import subprocess

import torch
import torch.distributed as dist
import pytest

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
    subprocess.check_call(['torchrun', '--nnodes', '1', '--nproc-per-node', '2', __file__, '--test-case', str(test_case)])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test-case', type=int, default=0)
    args = parser.parse_args()
    do_test_gather_and_split(args.test_case)
