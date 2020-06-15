'''
Module functional_bases.

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

import numpy as np
from copy import deepcopy
import tensap


class FunctionalBases:
    '''
    Class FunctionalBases.

    Attributes
    ----------
    bases : list or numpy.ndarray, or tensap.FunctionalBases, optional
        List or numpy.ndarray containing objects of type FunctionalBasis.
        The default is None.

    measure : tensap.Measure
        The measure associated with the functional bases. By default, a
        tensap.ProductMeasure constituted of the attribute measure of each
        basis of bases.

    '''

    def __init__(self, bases=None):
        '''
        Constructor for the class FunctionalBases.

        To create a FunctionalBases by replication of FunctionalBasis,
            see FunctionalBases.duplicate.

        Parameters
        ----------
        bases : list or numpy.ndarray, or tensap.FunctionalBases, optional
            List or numpy.ndarray containing objects of type FunctionalBasis.
            The default is None.

        Raises
        ------
        ValueError
            If the provided bases are not of type FunctionalBasis.

        Returns
        -------
        None.

        '''
        if bases is not None:
            if isinstance(bases, FunctionalBases):
                measure = bases.measure
                bases = bases.bases
            elif np.all([isinstance(x, tensap.FunctionalBasis) for
                         x in bases]):
                measure = tensap.ProductMeasure([x.measure for x in bases])
            else:
                raise ValueError('Wrong argument.')

        self.bases = np.atleast_1d(bases)
        self.measure = measure
        if not np.all([isinstance(x, tensap.FunctionalBasis)
                       for x in self.bases]):
            raise ValueError(
                'Bases must contain objects of type FunctionalBasis')

    def __repr__(self):
        return ('<{}:{n}' +
                '{t}bases = {},{n}' +
                '{t}measure = {}>').format(self.__class__.__name__,
                                           self.bases,
                                           self.measure,
                                           t='\t', n='\n')

    def length(self):
        '''
        Return the number of bases in self.

        Returns
        -------
        int
            The number of bases in self.

        '''
        return len(self.bases)

    def __len__(self):
        return self.length()

    def cardinals(self, ind=None):
        '''
        Return the number of functions in each basis of self, or in basis ind
        if provided.

        Parameters
        ----------
        ind : ind, optional
            The index of the selected basis. The default is None.

        Returns
        -------
        numpy.ndarray
            The number of functions in each basis of self, or in basis ind
            if provided.

        '''
        if ind is None:
            card = [x.cardinal() for x in self.bases]
        else:
            card = [x.cardinal() for x in self.bases[ind]]
        return np.array(card)

    def ndim(self):
        '''
        Return the dimension of each basis of self.

        Returns
        -------
        list
            The dimension of each basis of self.

        '''
        return [x.ndim() for x in self.bases]

    def domain(self):
        '''
        Return the domain of each basis of self.

        Returns
        -------
        list
            The domain of each basis of self.

        '''
        return [x.domain() for x in self.bases]

    def orthonormalize(self):
        '''
        Orthonormalize the basis functions of self.

        Returns
        -------
        tensap.FunctionalBases
            The FunctionalBases with orthonormalized basis functions.

        '''
        return FunctionalBases([x.orthonormalize() for x in self.bases])

    def kron(self, g):
        '''
        Return the bases obtained by the Kronecker product of two bases.

        Parameters
        ----------
        g : FunctionalBases
            The second bases of the product.

        Returns
        -------
        out : FunctionalBases
            The bases obtained by the Kronecker product of two bases.

        '''
        out = deepcopy(self)
        out.bases = [x.kron(y) for x, y in zip(self.bases, g.bases)]
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
        out : tensap.FunctionalBases
            The FunctionalBases with removed bases.

        '''
        out = deepcopy(self)
        out.bases = np.delete(self.bases, ind)
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
        out : tensap.FunctionalBases
            The FunctionalBases with kept bases.

        '''
        out = deepcopy(self)
        out.bases = out.bases[ind]
        return out

    def transpose(self, perm):
        '''
        Return self with the basis permutation perm.

        Parameters
        ----------
        perm : list or numpy.ndarray
            The permutation of the bases.

        Returns
        -------
        out : tensap.FunctionalBases
            The FunctionalBases with permuted bases.

        '''
        out = deepcopy(self)
        out.bases = out.bases[perm]
        return out

    def __eq__(self, bases_2):
        if not isinstance(bases_2, FunctionalBases):
            out = False
        else:
            out = np.all([x == y for x, y in zip(self.bases,
                                                 bases_2.bases)]) and \
                self.measure == bases_2.measure
        return out

    def adaptation_path(self):
        '''
        Compute the adaptationPath for each basis in self.

        See also tensap.FunctionalBasis.adaptation_path.

        Returns
        -------
        list
            List of adaptation paths for each basis.

        '''
        return np.array([x.adaptation_path() for x in self.bases],
                        dtype=object)

    def eval(self, x, dims=None, nargout=1):
        '''
        Computes evaluations of the basis functions of self at points x
        in dimensions dims if provided, in all the dimensions if not.

        Parameters
        ----------
        x : numpy.ndarray
            The input points.
        dims : list or numpy.ndarray, optional
            The dimensions of the bases to be evaluated. The default is None,
            indicating all the dimensions.
        nargout : int, optional
            Indicates the number of expected outputs. The default is 1,
            indicating to return only the evaluations of the basis functions.

        Returns
        -------
        out : list
            Evaluations of the basis functions of self.
        x : numpy.ndarray
            The input points, grouped by basis.

        '''
        if dims is None:
            dims = np.arange(self.length())
        if not isinstance(x, list):
            ind = np.atleast_1d(self.ndim())
            x = np.hsplit(x, np.cumsum(ind[dims])[:-1])

        out = [y.eval(z) for y, z in zip(self.bases[dims], x)]
        if nargout == 1:
            return out
        return out, x

    def eval_derivative(self, n, x, dims=None, nargout=1):
        '''
         Compute evaluations of the n-derivative of the basis functions of
         self at points x in each dimension in dims if provided, in all the
         dimensions otherwise.

        Parameters
        ----------
        n : list or numpy.ndarray
            The order of derivation in each dimension (in dims if provided).
        x : numpy.ndarray
            The input points.
        dims : list or numpy.ndarray, optional
            The dimensions of the bases for which the n-derivative is to be
            computed. The default is None, indicating all the dimensions.
        nargout : int, optional
            Indicates the number of expected outputs. The default is 1,
            indicating to return only the evaluations of the n-derivative of
            the basis functions.

        Returns
        -------
        out : list
            Evaluations of the n-derivative of the basis functions of self.
        x : numpy.ndarray
            The input points, grouped by basis.

        '''
        if dims is None:
            dims = np.arange(self.length())
        if not isinstance(x, list):
            ind = np.atleast_1d(self.ndim())
            x = np.hsplit(x, np.cumsum(ind[dims])[:-1])

        out = [y.eval_derivative(m, z) for
               y, m, z in zip(self.bases[dims], n, x)]
        if nargout == 1:
            return out
        return out, x

    def derivative(self, n):
        '''
        Compute the n-derivative of the basis functions of self.

        Parameters
        ----------
        n : list or numpy.ndarray
            The order of derivation in each dimension.

        Returns
        -------
        out : FunctionalBases
            A Functionalbases with the n-derivative of the basis functions of
            self.

        '''
        out = deepcopy(self)
        out.bases = [x.derivative(m) for x, m in zip(self.bases, n)]
        return out

    def random(self, *args, **kwargs):
        '''
        Compute random evaluations of the bases in self.

        Parameters
        ----------
        *args : misc
            Additional parameters for the random generation. See random_dims.

        Returns
        -------
        list or numpy.ndarray
            Random evaluations of the basis functions of self.
        numpy.ndarray
            The input points, grouped by basis.

        '''
        return self.random_dims(range(len(self)), *args, **kwargs)

    def random_dims(self, dims, n=1, measure=None, nargout=1):
        '''
        Evaluate the bases in dimensions dims of the bases of self
        using n points drawn randomly according to measure if provided, or to
        self.measure.marginal(dims) otherwise.

        Parameters
        ----------
        dims : list or numpy.ndarray
            The dimensions of the bases to be evaluated.
        n : int, optional
            The number of random evaluations. The default is 1.
        measure : tensap.ProbabilityMeasure, optional
            The probability measure used for the generation of the input
            points. The default is None, indicating to use
            self.measure.marginal(dims).

        Returns
        -------
        bases_eval : list or numpy.ndarray
            Random evaluations of the basis functions of self.
        x : numpy.ndarray
            The input points, grouped by basis.

        '''
        if measure is not None:
            assert isinstance(measure, tensap.ProbabilityMeasure), \
                'Must provide a ProbabilityMeasure.'
            x = measure.marginal(dims).random(int(n))
            bases_eval, x = self.eval(x, dims, 2)

            if len(bases_eval) == 1:
                bases_eval = bases_eval[0]

            if nargout == 1:
                return bases_eval
            return bases_eval, x
        return self.random_dims(dims, n, self.measure, nargout)

    def get_random_vector(self):
        '''
        Returns the random vector associated with self.

        Returns
        -------
        measure : tensap.RandomVector
            The random vector associated with self.

        '''
        if isinstance(self.measure, tensap.RandomVector):
            measure = self.measure
        elif isinstance(self.measure, tensap.ProbabilityMeasure):
            measure = self.measure.random_vector()
        else:
            measure = None
        return measure

    def one(self, dims=None):
        '''
        Return the coefficients associated with the FunctionalBases so that it
        returns one.

        Parameters
        ----------
        dims : list or numpy.ndarray, optional
            The dimensions of the bases that need to return one. The default
            is None, indicating all the bases.

        Returns
        -------
        list
            The list of coefficients so that the selected bases return one.

        '''
        if dims is None:
            dims = range(len(self))
        return [x.one() for x in self.bases[dims]]

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
        if dims is None:
            dims = range(len(self))
        return [x.gram_matrix() for x in self.bases[dims]]

    def storage(self):
        '''
        Return the storage requirement of the FunctionalBases.

        Returns
        -------
        int
            The storage requirement of the FunctionalBases.

        '''
        return np.sum([x.storage() for x in self.bases])

    def mean(self, dims=None, measure=None):
        '''
        Compute the mean of self in the dimensions in dims according to
        the RandomVector measure if provided, or to the standard RandomVector
        associated with each basis if not.

        Parameters
        ----------
        dims : list or numpy.ndarray, optional
            The dimensions of the bases for which the mean is to be computed.
            The default is None, indicating all the bases.
        measure : tensap.RandomVector or tensap.RandomVariable, optional
            The probability measure according to which the mean is computed.
            The default is None, indicating to use the self.measure.

        Returns
        -------
        out : list
            The mean of each basis functions.

        '''
        if dims is None:
            dims = np.arange(self.length())

        if isinstance(measure, tensap.RandomVector):
            out = [x.mean(y) for x, y in zip(self.bases[dims],
                                             measure.random_variables)]
        elif isinstance(measure, tensap.RandomVariable):
            out = [x.mean(measure) for x in self.bases[dims]]
        else:
            out = [x.mean() for x in self.bases[dims]]
        return out

    def interpolation_points(self, x=None):
        '''
        Return the interpolation points for the bases.

        Parameters
        ----------
        x : tensap.FullTensorGrid or list or numpy.ndarray
            The set of points in which the interpolation points are selected.

        Returns
        -------
        points : list
            The interpolation points.

        '''
        if x is not None:
            if isinstance(x, tensap.FullTensorGrid):
                x = x.grids
            elif not isinstance(x, list):
                x = np.split(x, x.shape[1], 1)

        points = []
        for k, basis in enumerate(self.bases):
            if x is None:
                points.append(basis.interpolation_points())
            else:
                points.append(basis.interpolation_points(x[k]))
        return points

    def tensor_product_interpolation(self, *args):
        '''
        Interpolate a function on a product grid.

        See also
        tensap.FullTensorProductFunctionalBasis.tensorProductInterpolation.

        Parameters
        ----------
        *args : tuple
            Parameters of the method tensorProductInterpolation of
            tensap.FullTensorProductFunctionalBasis.

        Returns
        -------
        tensap.FunctionalTensor
            The interpolation of the function on a product grid.

        '''
        H = tensap.FullTensorProductFunctionalBasis(self)
        return H.tensor_product_interpolation(*args)

    def magic_points(self, x, J=None):
        '''
        Provide the magic points associated with the functional bases selected
        in a given set of points x.

        Parameters
        ----------
        x : tensap.FullTensorGrid or list or numpy.ndarray
            The set of points in which the magic points are selected.
        J : numpy.ndarray, optional
            The default is None. If not none, selected the magic indices with
            tensap.magic_indices(F[:, J], self.cardinal(), 'left')[0]

        Returns
        -------
        points : list
            The magic points associated with each basis of self.
        ind : list
            The locations of the magic points in x for each basis of self.

        '''
        if isinstance(x, tensap.FullTensorGrid):
            x = x.grids
        elif not isinstance(x, list):
            x = np.split(x, x.shape[1], 1)

        points = [[]]*len(self)
        ind = [[]]*len(self)
        for k, basis in enumerate(self.bases):
            if J is None:
                points[k], ind[k], _ = basis.magic_points(x[k])
            else:
                points[k], ind[k], _ = basis.magic_points(x[k], J[k])
        return points, ind

    @staticmethod
    def duplicate(basis, dim):
        '''
        Create a FunctionalBases with bases created with a duplication of
        basis d times.

        Parameters
        ----------
        basis : tensap.FunctionalBasis
            The basis to be duplicated.
        dim : int
            The number of times basis is duplicated.

        Returns
        -------
        tensap.FunctionalBases
            The obtained FunctionalBases.

        '''
        return FunctionalBases([basis]*dim)
