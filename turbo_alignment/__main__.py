import os
from turbo_alignment.cli.app import app


def set_prctl():
    try:
        import prctl

        prctl.set_ptracer(prctl.SET_PTRACER_ANY)

    except ImportError:
        print('prctl unavailable')


if __name__ == '__main__':
    set_prctl()
    os.register_at_fork(after_in_child=set_prctl)
    app()

# app()
