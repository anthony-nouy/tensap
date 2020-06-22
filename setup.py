'''
Setup file for tensap (tensor approximation package).

Copyright (c) 2020, Anthony Nouy, Erwan Grelier
This file is part of tensap (tensor approximation package).

tensap is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

tensap is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with tensap.  If not, see <https://www.gnu.org/licenses/>.

'''

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tensap",
    version="1.1",
    author="Anthony Nouy, Erwan Grelier",
    author_email="anthony.nouy@ec-nantes.fr",
    description="Tensor Approximation Package: a Python package for the approximation of functions and tensors.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://anthony-nouy.github.io/tensap/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'scipy',
    ],
    python_requires='>=3.6',
    include_package_data=True,
)
