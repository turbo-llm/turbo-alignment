import argparse
import pathlib

import numpy as np

from tests.sequence_parallel.utils import (
    ComparisonError,
    get_shape_suffix,
    read_first_line,
)


def compare(root_dir: pathlib.Path):
    failed = []
    for file in root_dir.glob('*.shape'):
        if '_rank_' in file.name:
            continue

        print('Process file', file.name)
        param_name = file.name[:-len('.shape')]
        print('Param name', param_name)

        all_shapes = set()
        for rank in (None, 0, 1):
            rank_file = file.with_name(param_name + get_shape_suffix(rank) + '.shape')
            shape = tuple(map(int, read_first_line(rank_file).strip().split(' ')))
            all_shapes.add(shape)
            print(f'{param_name=} {rank=} {shape=}')
        if len(all_shapes) != 1:
            raise ValueError('Found different shapes!')

        weights = {}
        canon_shape = next(iter(all_shapes))
        for rank in (None, 0, 1):
            grad_file = file.with_name(param_name + get_shape_suffix(rank) + '.npy')
            print(f'read weight from {grad_file}')

            weight = np.fromfile(grad_file, dtype=np.float32).reshape(canon_shape)
            # assert weight.shape == canon_shape, (weight.shape, canon_shape)
            weights[rank] = weight

        # combined_weights = weights[0] / 2 + weights[1] / 2
        combined_weights = weights[0] + weights[1]
        try:
            # np.testing.assert_allclose(weights[None], combined_weights, atol=1e-4, rtol=0.2)
            np.testing.assert_allclose(weights[None], combined_weights, atol=1e-3, rtol=0.2)
        except AssertionError as e:
            print(e.args)
            failed.append(param_name)
        else:
            print('Check done!')

        # break

        del weights, combined_weights

    if failed:
        raise ComparisonError(failed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=pathlib.Path, default=pathlib.Path('/mnt/p.geyn/forward'))
    args = parser.parse_args()
    compare(args.root_dir)
