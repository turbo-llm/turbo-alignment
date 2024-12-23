import os
import torch
import torch.distributed as dist


def create_hook(name: str, output_dir: str = '/mnt/p.geyn/gradients', seq_p_inited: bool = False):
    def hook(grad):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        middle_name = f'_rank_{dist.get_rank()}' if seq_p_inited else ''
        shape_filename = os.path.join(output_dir, name + middle_name + '.shape')
        filename = os.path.join(output_dir, name + middle_name + '.npy.gz')

        with open(shape_filename, 'w') as output:
            output.write(' '.join(map(str, grad.shape)))
            output.write('\n')
            output.write(str(grad.dtype))

        print(f'{dist.get_rank()=} {name=} {grad.min().item()=} {grad.max().item()=}')

        np_grad = grad.float().cpu().numpy()
        print(f'{dist.get_rank()=} {name=} {np_grad.min().item()=} {np_grad.max().item()=}')
        np_grad.tofile(filename)

    return hook


def write_shape(ar, filename):
    with open(filename, 'w') as output:
        output.write(' '.join(map(str, ar.shape)))
        output.write('\n')
        output.write(str(ar.dtype))


import functools

@functools.lru_cache()
def print_once(msg):
    print(msg)


def process_forward_output(output_dir, name, middle_name, output, module=None, args=None):
    shape_filename = os.path.join(output_dir, name + middle_name + '.shape')
    filename = os.path.join(output_dir, name + middle_name + '.npy.gz')
    if os.path.exists(filename):
        return

    if not isinstance(output, torch.Tensor):
        print_once(f'have output: {type(output)}')
        return

    write_shape(output, shape_filename)

    print(f'{dist.get_rank()=} {name=} {output.min().item()=} {output.max().item()=}')

    np_output = output.float().cpu().detach().numpy()
    print(f'{dist.get_rank()=} {name=} {np_output.min().item()=} {np_output.max().item()=}')
    np_output.tofile(filename)

    if name == 'model.layers.1.input_layernorm':
        filename = os.path.join(output_dir, name + '_weight_' + middle_name + '.npy')
        shape_filename = os.path.join(output_dir, name + '_weight_' + middle_name + '.shape')
        if os.path.exists(filename):
            return

        write_shape(module.weight, shape_filename)
        weight = module.weight.float().detach().cpu().numpy()
        weight.tofile(filename)
        print(args[0])


def create_forward_hook(name: str, output_dir: str = '/mnt/p.geyn/forward', seq_p_inited: bool = False):
    def hook(module, args, output):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        middle_name = f'_rank_{dist.get_rank()}' if seq_p_inited else ''
        if isinstance(output, tuple):
            for ind, local_output in enumerate(output):
                process_forward_output(output_dir, f'{name}_{ind}', middle_name, local_output, module, args)
        else:
            process_forward_output(output_dir, f'{name}', middle_name, output, module, args)

    return hook


