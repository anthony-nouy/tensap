# Copyright (c) 2020, Anthony Nouy, Erwan Grelier
# This file is part of tensap (tensor approximation package).

# tensap is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# tensap is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with tensap.  If not, see <https://www.gnu.org/licenses/>.


"""
Module interpolation_points_feature_map.

"""
import tensap


def interpolation_points_feature_map(F, x):
    """
    Constructs interpolation points for a feature map or functional basis
    using a greedy algorithm.

    Parameters
    ----------
    F : FunctionalBasis
        A functional basis of cardinal m defined on R^d, with an `eval` method.
    x : numpy.ndarray
        An N-by-d array where interpolation points are selected as n rows of x.

    Returns
    -------
    xI : numpy.ndarray
        A m-by-d array containing the selected interpolation points.
    """

    # Evaluate the functional basis F at points x
    F_eval = F.eval(x)

    # Apply the greedy algorithm to select points (assuming greedyAlgorithml2 is defined)
    # Transpose to match the MATLAB behavior
    L = tensap.greedy_algorithml2(F_eval.T, [])

    # Select the corresponding rows from x
    xI = x[L, :]

    return xI
