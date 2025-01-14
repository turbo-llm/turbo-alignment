import functools

import torch.distributed as dist


def run_in_order(group: dist.ProcessGroup | None = None):
    def inner(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            rank = dist.get_rank(group)
            for i in range(dist.get_world_size(group)):
                if i == rank:
                    res = f(*args, **kwargs)

                dist.barrier(group)

            return res

        return wrapped

    return inner
