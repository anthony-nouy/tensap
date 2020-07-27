import glob
import pytest
import time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def execfile(filepath, globals=None, locals=None):
    if globals is None:
        globals = {}
    globals.update({
        "__file__": filepath,
        "__name__": "__main__",
    })
    with open(filepath, 'rb') as file:
        exec(compile(file.read(), filepath, 'exec'), globals, locals)


@pytest.mark.parametrize("f", glob.glob('tutorials/**/*.py', recursive=True))
def test_tutorial(f):
    print(f"-- running {f}", flush=True)
    t0 = time.time()
    execfile(f)
    dt = time.time() - t0
    print(f"-- {f} ran in {dt:.2f} s", flush=True)
    plt.close("all")
