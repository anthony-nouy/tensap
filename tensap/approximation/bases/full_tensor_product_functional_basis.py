'''
Module full_tensor_product_functional_basis.

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
import tensap


class FullTensorProductFunctionalBasis(tensap.FunctionalBasis):
    '''
    Class FullTensorProductFunctionalBasis.

    Attributes
    ----------
    bases : list or tensap.FunctionalBases
        The bases associated with the object.

    '''

    def __init__(self, bases):
        '''
        Constructor for the class FullTensorProductFunctionalBasis.

        Parameters
        ----------
        bases : list or tensap.FunctionalBases
            The bases associated with the object.

        Returns
        -------
        None.

        '''
        tensap.FunctionalBasis.__init__(self)

        if isinstance(bases, list):
            bases = tensap.FunctionalBases(bases)

        assert isinstance(bases, tensap.FunctionalBases), \
            'The first argument must be a FunctionalBases.'

        self.bases = bases
        self.measure = tensap.ProductMeasure([x.measure for x in bases.bases])
        self.is_orthonormal = np.all([x.is_orthonormal for x in bases.bases])

    def __eq__(self, G):
        if not isinstance(G, FullTensorProductFunctionalBasis):
            out = False
        else:
            out = self.bases == G.bases
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
        return np.prod([x.cardinal() for x in self.bases.bases])

    def ndim(self):
        return np.sum(self.bases.ndim())

    def domain(self):
        return self.bases.domain()

    def orthonormalize(self):
        out = deepcopy(self)
        out.bases = out.bases.orthonormalize()
        return out

    def remove_bases(self, ind):
        '''
        Remove bases of self of index ind.

        Parameters
        ----------
        ind : int or list or numpy.ndarray
            The indices of the bases to remove.

        Returns
        -------
        tensap.FullTensorProductFunctionalBasis
            The FullTensorProductFunctionalBasis with removed bases.

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
        tensap.FullTensorProductFunctionalBasis
            The FullTensorProductFunctionalBasis with kept bases.

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
        tensap.FullTensorProductFunctionalBasis
            The FullTensorProductFunctionalBasis with permuted bases.

        '''
        out = deepcopy(self)
        out.bases = out.bases.transpose(perm)
        return out

    def mean(self, *args):
        H = self.bases.mean(None, *args)
        H = tensap.CanonicalTensor([np.reshape(x, [-1, 1]) for x in H], [1])
        return np.ravel(H.full().numpy())

    def conditional_expectation(self, dims, *args):
        ind = tensap.MultiIndices.bounded_by(self.bases.cardinals()-1)
        h = tensap.SparseTensorProductFunctionalBasis(self.bases, ind)
        return h.conditional_expectation(dims, *args)

    def eval(self, x):
        Hx = self.bases.eval(x)
        ind = tensap.MultiIndices.bounded_by(self.bases.cardinals()-1)
        y = Hx[0][:, ind.array[:, 0]]
        for i in np.arange(1, len(Hx)):
            y *= Hx[i][:, ind.array[:, i]]
        return y

    def gram_matrix(self, dims=None):
        '''
        Return the gram matrix of each basis of self, or of a selection of
        them if dims is provided.

        Parameters
        ----------
        dims : list or numpy.ndarray, optional
            The dimensions of the bases for which the gram matrix is computed.
            The default is None, indicating all the bases.

        Returns
        -------
        list
            The gram matrix of the selected bases.

        '''
        return self.bases.gram_matrix(dims)

    def projection(self, fun, I):
        '''
        Compute the projection of the function fun on the basis functions of
        self.

        Parameters
        ----------
        fun : tensap.Function
            The function to project.
        I : tensap.IntegrationRule
            The integration rule used to compute the projection.

        Raises
        ------
        NotImplementedError
            If the provided integration rule is not a
            tensap.FullTensorProductIntegrationRule.

        Returns
        -------
        tensap.FunctionalTensor
            The projection of the function fun on the basis functions of self.
        output : dict
            Dictionnary containing the number of evaluations of the function in
            the key 'number_of_evaluations'.

        '''
        if not isinstance(I, tensap.FullTensorProductIntegrationRule):
            raise NotImplementedError('Method not implemented.')

        u = fun.eval_on_tensor_grid(I.points)
        output = {'number_of_evaluations': u.storage()}
        Hx = self.bases.eval(np.hstack(I.points.grids))
        Mx = [np.matmul(np.transpose(x), np.matmul(np.diag(y), x)) for
              x, y in zip(Hx, I.weights)]
        Hx_w = [np.linalg.solve(m, np.matmul(np.transpose(x), np.diag(y))) for
                m, x, y in zip(Mx, Hx, I.weights)]
        f_dims = np.arange(len(Hx_w))
        u_coeff = u.tensor_matrix_product(Hx_w, f_dims)
        return tensap.FunctionalTensor(u_coeff, self.bases, f_dims), output

    def optimal_sampling_measure(self):
        # TODO optimal_sampling_measure
        raise NotImplementedError('Method not implemented.')

    def interpolation_points(self, *args):
        grid = self.bases.interpolation_points(*args)
        points = tensap.FullTensorGrid(grid).array()
        return points, grid

    def tensor_product_interpolation(self, fun, grid=None):
        '''
        Return the interpolation of function fun on a product grid.

        Parameters
        ----------
        fun : function or tensap.Function or tensap.Tensor
            The function to interpolate, or a tensor of order d whose entries
            are the evaluations of the function on a product grid.
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
        if grid is None:
            grid = self.bases.interpolation_points()
        else:
            grid = self.bases.interpolation_points(grid)
        grid = tensap.FullTensorGrid(grid)

        output = {}
        if hasattr(fun, '__call__') or isinstance(fun, tensap.Function):
            x_grid = grid.array()
            y = fun(x_grid)
            y = tensap.FullTensor(y, grid.dim, grid.shape)
            output['number_of_evaluations'] = y.storage()
        # TODO Create an empty class Tensor for when using isinstance?
        elif isinstance(fun, (tensap.FullTensor, tensap.CanonicalTensor,
                              tensap.TreeBasedTensor), tensap.DiagonalTensor):
            y = fun
        else:
            raise ValueError('The argument fun should be a Function, ' +
                             'function, or a Tensor.')
        output['grid'] = grid

        B = self.bases.eval(grid.grids)
        B = [np.linalg.inv(x) for x in B]
        y = y.tensor_matrix_product(B)
        return tensap.FunctionalTensor(y, self.bases), output
