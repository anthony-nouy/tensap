'''
Module sparse_tensor_product_functional_basis.

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

from copy import deepcopy
import numpy as np
from scipy.sparse import eye as speye
import tensap


class SparseTensorProductFunctionalBasis(tensap.FunctionalBasis):
    '''
    Class SparseTensorProductFunctionalBasis.

    Attributes
    ----------
    bases : list or tensap.FunctionalBases
        The bases associated with the object.
    indices : tensap.MultiIndices
        The indices of the basis functions (the indices start at 0).

    '''

    def __init__(self, bases, indices):
        '''
        Constructor for the class SparseTensorProductFunctionalBasis.

        Parameters
        ----------
        bases : list or tensap.FunctionalBases
            The bases associated with the object.
        indices : tensap.MultiIndices
            The indices of the basis functions (the indices start at 0).

        Returns
        -------
        None.

        '''
        tensap.FunctionalBasis.__init__(self)

        assert isinstance(bases, tensap.FunctionalBases), \
            'The first argument must be a FunctionalBases.'

        assert isinstance(indices, tensap.MultiIndices), \
            'The second argument must be a MultiIndices.'

        self.bases = bases
        self.indices = indices
        self.measure = tensap.ProductMeasure([x.measure for x in bases.bases])
        self.is_orthonormal = np.all([x.is_orthonormal for x in bases.bases])

    def __eq__(self, G):
        if not isinstance(G, SparseTensorProductFunctionalBasis):
            out = False
        else:
            out = np.all(self.bases == G.bases) and \
                np.all(self.indices == G.indices)
        return out

    def length(self):
        '''
        Return the number of bases in self.bases.

        Returns
        -------
        int
            The number of bases in self.bases.

        '''
        return len(self.bases)

    def __len__(self):
        return self.length()

    def cardinal(self):
        return self.indices.cardinal()

    def ndim(self):
        return self.indices.ndim()

    def domain(self):
        return self.bases.domain()

    def remove_bases(self, ind):
        '''
        Remove bases of self of index ind.

        Parameters
        ----------
        ind : int or list or numpy.ndarray
            The indices of the bases to remove.

        Returns
        -------
        tensap.SparseTensorProductFunctionalBasis
            The SparseTensorProductFunctionalBasis with removed bases.

        '''
        out = deepcopy(self)
        out.bases = out.bases.remove_bases(ind)
        return out

    def keep_bases(self, ind):
        '''
        Keep only the bases of self of index ind.

        Parameters
        ----------
        ind : int or list or numpy.ndarray
            The indices of the bases to keep.

        Returns
        -------
        tensap.SparseTensorProductFunctionalBasis
            The SparseTensorProductFunctionalBasis with kept bases.

        '''
        out = deepcopy(self)
        out.bases = out.bases.keep_bases(ind)
        return out

    def remove_mapping(self, ind):
        # TODO remove_mapping
        return self

    def keep_mapping(self, ind):
        # TODO keep_mapping
        return self

    def transpose(self, perm):
        '''
        Return self with the basis permutation perm.

        Parameters
        ----------
        perm : list or numpy.ndarray
            The permutation of the bases.

        Returns
        -------
        tensap.SparseTensorProductFunctionalBasis
            The SparseTensorProductFunctionalBasis with permuted bases.

        '''
        out = deepcopy(self)
        out.bases = out.bases.transpose(perm)
        return out

    def mean(self, *args):
        M = self.bases.mean(None, *args)
        m = M[0][self.indices.array[:, 0]]
        for i in np.arange(1, len(M)):
            m *= M[i][self.indices.array[:, i]]
        return m

    def conditional_expectation(self, dims, *args):
        dims = np.atleast_1d(dims)
        if np.all([isinstance(x, bool) for x in dims]):
            dims = np.nonzero(dims)[0]

        dims_C = np.setdiff1d(range(self.ndim()), dims)

        if dims_C.size != 0:
            M = self.bases.mean(dims_C, *args)
            I = self.indices
            J = I.keep_dims(dims)
            m = M[0][I.array[:, dims_C[0]]]
            for i in np.arange(1, len(M)):
                m *= M[i][I.array[:, dims_C[i]]]

            if dims.size == 0:
                return m

            ind = np.nonzero(np.all(J.array == I.array[:, dims][:, np.newaxis],
                                    axis=2))[1]

            d = np.zeros((J.cardinal(), I.cardinal()))
            d[ind, range(I.cardinal())] = m

            h = self.keep_bases(dims)
            # TODO Uncomment when the mappings are implemented
            # h = self.keep_mapping(dims)
            h.indices = J
            h = tensap.FunctionalBasisArray(d, h, I.cardinal())

        else:
            h = tensap.FunctionalBasisArray(np.eye(self.cardinal()), self,
                                            [self.cardinal(), 1])
        return h

    def get_random_vector(self):
        '''
        Return the random vector associated with the basis functions of self.

        Returns
        -------
        tensap.RandomVector
            The random vector associated with the basis functions of self.

        '''
        return self.bases.get_random_vector()

    def eval(self, x):
        Hx = self.bases.eval(x)
        return self.eval_with_functional_bases_evals(Hx)

    def eval_with_functional_bases_evals(self, Hx):
        y = Hx[0][:, self.indices.array[:, 0]]
        for i in np.arange(1, len(Hx)):
            y *= Hx[i][:, self.indices.array[:, i]]
        return y

    def eval_derivative(self, n, x):
        dnHx = self.bases.eval_derivative(n, x)
        return self.eval_with_functional_bases_evals(dnHx)

    def derivative(self, n):
        out = deepcopy(self)
        out.bases = out.bases.derivative(n)
        return out

    def adaptation_path(self, p=1):
        '''
        Create an adaptation path associated with increasing p-norm of
        multi-indices.

        Parameters
        ----------
        p : float, optional
            The positive real scalar p of the p-norm. The default is 1.

        Returns
        -------
        P : numpy.ndarray
            The adaptation path.

        '''
        n = self.indices.norm(p)
        n_unique = np.sort(np.unique(n))
        P = np.full((self.indices.cardinal(), n_unique.size), False)
        for k in range(n_unique.size):
            P[n <= n_unique[k], k] = True
        return P

    def gram_matrix(self,):
        '''
        Return the gram matrix of the basis.

        Returns
        -------
        numpy.ndarray
            The gram matrix of the basis.

        '''
        if self.is_orthonormal:
            M = speye(self.indices.cardinal())
        else:
            G = self.bases.gram_matrix()
            ind = self.indices.array
            M = G[0][np.ix_(ind[:, 0], ind[:, 0])]
            for i in np.arange(1, ind.shape[1]):
                M *= G[i][np.ix_(ind[:, i], ind[:, i])]
        return M

    def plot_multi_indices(self, *args):
        '''
        PLot the multi-index set of the object.

        See also tensap.MultiIndices.plot.

        Parameters
        ----------
        *args : tuple
            Additional parameters for tensap.MultiIndices' plot method.

        Returns
        -------
        None.

        '''
        self.indices.plot(*args)

    def tensor_product_interpolation(self, fun, *args):
        '''
        Return the interpolation of function fun on a sparse grid.

        Parameters
        ----------
        fun : function or tensap.Function
            The function to interpolate.
        grid : list, optional
            The grid of points used for the interpolation. If one grid has more
            points than the dimension of the corresponding basis, use
            magicPoints for the selection of a subset of points adapted to the
            basis. The default is None, indicating to use the method
            self.bases.interpolation_points().

        Raises
        ------
        ValueError
            If the argument fun is neither a tensap.Function, a function nor
            a tensap.Tensor.

        Returns
        -------
        tensap.FunctionalTensor
            The interpolation of the function.
        output : dict
            A dictionnary of outputs of the method.

        '''
        grid = self.bases.interpolation_points(*args)
        grid = tensap.SparseTensorGrid(grid, self.indices)
        x_grid = grid.array()
        f_interp = self.interpolate(fun, x_grid)
        output = {'number_of_evaluations': x_grid.shape[0], 'grid': grid}

        return f_interp, output
