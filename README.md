[![Build Status](https://github.com/anthony-nouy/tensap/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/anthony-nouy/tensap/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/anthony-nouy/tensap/branch/master/graph/badge.svg)](https://codecov.io/gh/anthony-nouy/tensap)
[![Doc](https://readthedocs.org/projects/spack/badge/?version=latest)](https://anthony-nouy.github.io/sphinx/tensap/master/)


# tensap (Tensor Approximation Package)

A Python package for the approximation of functions and tensors.

tensap features low-rank tensors (including canonical, tensor-train and tree-based tensor formats or tree tensor networks), sparse tensors, polynomials, and allows the plug-in of other approximation tools. It provides different approximation methods based on interpolation, least-squares projection or statistical learning.

See the tutorials folder and its subfolders for examples of use.

Install from PyPi:

```
pip install tensap
```


Install from conda:

```
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install tensap
```
