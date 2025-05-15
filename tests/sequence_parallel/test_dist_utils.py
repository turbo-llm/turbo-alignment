import subprocess

import torch
import torch.distributed as dist
import pytest

from turbo_alignment.dist_utils.comm import create_and_broadcast
from turbo_alignment.dist_utils.gather_and_split import all_gather_variable, gather_and_split

from tests.sequence_parallel.marks import has_two_gpus

MODE_TO_FUNCTION = {}


def register_function(name):
    def inner(f):
        MODE_TO_FUNCTION[name] = f
        return f

    return inner


INPUTS = [
    [[1, 2], [3]],
    [[1, 2, 3], [4, 5, 6]],
]

OUTPUTS = [
    [[0, 1], [2, 3]],
    [[1, 2, 3], [4, 5, 6]],
]


assert len(INPUTS) == len(OUTPUTS)


def create_run_preamble(num_gpus: int):
    return ['torchrun', '--nnodes', '1', '--nproc-per-node', str(num_gpus), __file__]


@register_function('all_gather_variable')
def do_test_all_gather_variable(n_ranks: int = 4, group_size: int = 2):
    assert n_ranks % group_size == 0, (n_ranks, group_size)

    dist.init_process_group()
    rank = dist.get_rank()

    group_count = n_ranks // group_size

    device = torch.device('cuda', index=rank)

    inputs_per_rank = [torch.tensor([i] * (i + 1)) for i in range(n_ranks)]
    print(rank, n_ranks, inputs_per_rank)

    group = None
    for group_id in range(group_count):
        ranks = [group_size * group_id + i for i in range(group_size)]
        local_group = dist.new_group(ranks=ranks)
        if rank in ranks:
            group = local_group

    assert group is not None

    input_on_rank = inputs_per_rank[rank].to(device=device)
    result = torch.cat(all_gather_variable(input_on_rank, group=group), dim=-1)
    assert result.tolist() == sum((inputs_per_rank[r].tolist() for r in dist.get_process_group_ranks(group)), [])


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason='at least four gpus required',
)
def test_all_gather_variable():
    return subprocess.check_call(create_run_preamble(4) + ['--mode', 'all_gather_variable'])


@register_function('gather_and_split')
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
    subprocess.check_call(create_run_preamble(2) + ['--mode', 'gather_and_split', '--test-case', str(test_case)])


CREATE_AND_BROADCAST_INPUTS = [
    [1, 2, 3],
    [True, False, True, False],
]


@register_function('create_and_broadcast')
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


@pytest.mark.skipif(not has_two_gpus(), reason='at least two gpus are required')
@pytest.mark.parametrize(
    'test_case',
    range(len(CREATE_AND_BROADCAST_INPUTS)),
)
def test_create_and_broadcast(test_case: int):
    subprocess.check_call(create_run_preamble(2) + ['--mode', 'create_and_broadcast', '--test-case', str(test_case)])


if __name__ == '__main__':
    import argparse
    import inspect

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=list(MODE_TO_FUNCTION.keys()))
    parser.add_argument('--test-case', type=int, default=0)
    args = parser.parse_args()

    fun = MODE_TO_FUNCTION[args.mode]
    kwargs = {}
    if 'test_case' in inspect.signature(fun).parameters:
        kwargs['test_case'] = args.test_case

    fun(**kwargs)
