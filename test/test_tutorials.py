import subprocess
import sys
import glob
import pytest
import os
import time

@pytest.mark.parametrize("f", glob.glob('tutorials/**/*.py', recursive=True))
def test_tutorial(f):
    print(f"-- running {f}")
    mpl_env = os.environ.copy()
    mpl_env["MPLBACKEND"] = "agg"
    t0 = time.time()
    subprocess.run([sys.executable, f], check=True, env=mpl_env)
    dt = time.time() - t0
    print(f"-- {f} ran in {dt:.2f} s")
