# tensap (Tensor Approximation Package)

A Python package for the approximation of functions and tensors. 

tensap features low-rank tensors (including canonical, tensor-train and tree-based tensor formats or tree tensor networks), sparse tensors, polynomials, and allows the plug-in of other approximation tools. It provides different approximation methods based on interpolation, least-squares projection or statistical learning.

See the tutorials folder and its subfolders for examples of use.



To install tensap directly from github, run

```
pip install git+git://github.com/anthony-nouy/tensap@master
```

Alternatively, you can `git clone` (if you do not have a local version) or `git pull` the repository, and then run from the tensap folder containing the file setup.py

```
pip install .
```

If an older version of tensap if already installed, you may need to uninstall it prior to installing the newer version.