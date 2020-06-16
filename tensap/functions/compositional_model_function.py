'''
Module compositional_model_function

Copyright (c) 2020, Anthony Nouy, Erwan Grelier
This file is part of tensap (tensor approximation package).

tensap is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

tensap is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with tensap.  If not, see <https://www.gnu.org/licenses/>.

'''

import sys
import numpy as np
sys.path.insert(0, './../../../')
import tensap


class CompositionalModelFunction(tensap.Function):
    '''
    Class CompositionalModelFunction.

    Attributes
    ----------
    tree : tensap.DimensionTree
        The dimension tree associated with the function.
    fun : list or numpy.ndarray or function or tensap.Function
        A list or array of functions, one for each node of the tree, or
        one function, identical for all the internal nodes of the tree.
    measure : tensap.Measure
        The measure associated with the function.

    '''

    def __init__(self, tree, fun, measure):
        '''
        Constructor for the class CompositionalModelFunction.

        Parameters
        ----------
        tree : tensap.DimensionTree
            The dimension tree associated with the function.
        fun : list or numpy.ndarray or function or tensap.Function
            A list or array of functions, one for each node of the tree, or
            one function, identical for all the internal nodes of the tree.
        measure : tensap.Measure
            The measure associated with the function.

        Returns
        -------
        None.

        '''
        tensap.Function.__init__(self)

        self.tree = tree

        if not isinstance(fun, (list, np.ndarray)):
            self.fun = np.empty(tree.nb_nodes, dtype=object)
            self.fun[tree.internal_nodes-1] = fun
        else:
            self.fun = np.array(fun, dtype=object)

        self.dim = tree.dim2ind.size
        self.measure = measure

    def eval(self, x):
        '''
        Evaluate the CompositionalModelFunction at points x.

        Parameters
        ----------
        x : list or numpy.ndarray
            The points at which the function is to be evaluated.

        Returns
        -------
        numpy.ndarray
            The evaluations of the function at points x.

        '''
        x = np.atleast_2d(x)
        tree = self.tree
        z = np.empty(tree.nb_nodes, dtype=object)
        for nu in range(self.dim):
            z[tree.dim2ind[nu]-1] = x[:, nu]

        for level in np.arange(np.max(tree.level), -1, -1):
            nodes = tree.nodes_with_level(level)
            for nod in tensap.fast_setdiff(nodes, tree.dim2ind):
                ch = tree.children(nod)
                z[nod-1] = self.fun[nod-1](*z[ch-1])

        return z[tree.root-1]
