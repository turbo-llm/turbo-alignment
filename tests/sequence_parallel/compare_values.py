import argparse
import pathlib

import numpy as np

from tests.sequence_parallel.utils import (
    ComparisonError,
    get_shape_suffix,
    read_first_line,
)


def compare(root_dir, attention_mask = None):
    module_file = root_dir / 'forward_order.txt'
    failed = []
    with module_file.open('r', encoding='utf-8') as input_:
        for param_name in input_:
            param_name = param_name.strip()
            if param_name.endswith('_1') or param_name.endswith('_2'):
                continue

            half_shape = None
            full_shape = None
            for rank in (None, 0, 1):
                rank_file = root_dir / (param_name + get_shape_suffix(rank) + '.shape')
                shape = tuple(map(int, read_first_line(rank_file).strip().split(' ')))
                if rank is None:
                    full_shape = shape

                else:
                    if half_shape is not None:
                        if shape != half_shape:
                            raise ValueError(f'Shaped mismatched: {shape=} {half_shape}')
                    else:
                        half_shape = shape

                print(f'{param_name=} {rank=} {shape=}')

            weights = {}
            for rank in (None, 0, 1):
                grad_file = root_dir / (param_name + get_shape_suffix(rank) + '.npy')
                print(f'read weight from {grad_file}')

                canon_shape = full_shape if rank is None else half_shape
                weight = np.fromfile(grad_file, dtype=np.float32).reshape(canon_shape)
                # assert weight.shape == canon_shape, (weight.shape, canon_shape)
                weights[rank] = weight

            if 'rotary_emb' in param_name:
                np.testing.assert_allclose(weights[0], weights[1])
                combined_weights = weights[0]

            else:
                if len(half_shape) == 3:
                    dim_to_merge = 1
                else:
                    raise ValueError(f'Cannot merge: {half_shape}')

                combined_weights = np.concatenate([weights[0], weights[1]], axis=dim_to_merge)
            try:
                # if attention_mask is None or 'rotary_emb' in param_name:
                if attention_mask is None:
                    np.testing.assert_allclose(weights[None], combined_weights, atol=0.01, rtol=0.01)

                else:
                    if 'rotary_emb' not in param_name:
                        assert combined_weights.shape[0] == attention_mask.shape[0], (combined_weights.shape[0], attention_mask.shape[0])

                        for i in range(combined_weights.shape[0]):
                            attention_row = attention_mask[i]
                            np.testing.assert_allclose(
                                weights[None][i, attention_row],
                                combined_weights[i, attention_row],
                                atol=0.01,
                                rtol=0.01,
                            )

                    else:
                        assert combined_weights.shape[0] == 1, f'Unexpected shape in rotary_emb: {combined_weights.shape}'
                        for i in range(combined_weights.shape[0]):
                            attention_row = attention_mask[i]
                            np.testing.assert_allclose(
                                weights[None][0, attention_row],
                                combined_weights[i, attention_row],
                                atol=0.01,
                                rtol=0.01,
                            )

            except AssertionError as e:
                print(e.args[0])
                failed.append(param_name)
            else:
                print('Check done!')

            del weights, combined_weights

    if failed:
        raise ComparisonError(failed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=pathlib.Path, default=pathlib.Path('/mnt/p.geyn/forward'))
    args = parser.parse_args()
    compare(args.root_dir)
