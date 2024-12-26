import pathlib


def get_shape_suffix(rank: int | None) -> str:
    if rank is None:
        return ''

    return f'_rank_{rank}'


def read_first_line(p: pathlib.Path) -> str:
    with open(p, 'r') as input_:
        return next(iter(input_))


class ComparisonError(Exception):
    def __init__(self, failed: list[str]):
        super().__init__()
        self.failed = failed