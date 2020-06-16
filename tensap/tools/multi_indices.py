'''
Module multi_indices.

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
# import tensap


class MultiIndices:
    '''
    Class MultiIndices.

    Attributes
    ----------
    array : numpy.ndarray
        An array containing n multi-indices in N^d.

    '''

    def __init__(self, array=None):
        '''
        Constructor for the class MultiIndices.

        Parameters
        ----------
        array : list or numpy.ndarray or tensap.MultiIndices, optional
            The array characterizing the set of multi-indices. The default is
            None.

        Returns
        -------
        None.

        '''
        if hasattr(array, 'array'):
            array = array.array
        self.array = np.atleast_2d(array)

    def sub2ind(self, shape):
        '''
        Convert the indices of the MultiIndices object into flat indices.

        Parameters
        ----------
        shape : list or numpy.ndarray
            The shape of the array.

        Returns
        -------
        numpy.ndarray
            The flat indices associated with the MultiIndices object and shape.

        '''
        ind = self.to_list()
        return np.ravel_multi_index(ind, shape, order='F')

    def __eq__(self, J):
        if not isinstance(J, MultiIndices):
            ok = False
        else:
            if J.cardinal() == 1:
                J.array = np.tile(J.array, (self.cardinal(), 1))
            ok = np.all(self.array == J.array, axis=1)
        return ok

    def __le__(self, J):
        assert isinstance(J, MultiIndices), 'Must provide a MultiIndices.'
        if J.cardinal() == 1:
            J.array = np.tile(J.array, (self.cardinal(), 1))
        return np.all(self.array <= J.array, axis=1)

    def __add__(self, m):
        return MultiIndices(self.array+m)

    def __sub__(self, m):
        return MultiIndices(self.array-m)

    def cardinal(self):
        '''
        Return the cardinal of the MultiIndices object.

        Returns
        -------
        int
            The cardinal of the MultiIndices object.

        '''
        return self.array.shape[0]

    def to_list(self):
        '''
        Convert the MultiIndices' array into a list of arrays.

        Returns
        -------
        list
            The MultiIndices' array as a list of arrays.

        '''
        return [self.array[:, i] for i in range(self.ndim())]

    def ndim(self):
        '''
        Return the dimension of the multi-indices.

        Returns
        -------
        int
            The dimension of the multi-indices.

        '''
        return self.array.shape[1]

    def norm(self, p=2, k=None):
        '''
        Compute the p-norm of multi-indices k in the object.

        Parameters
        ----------
        p : int or numpy.inf, optional
            The positive real scalar p of the p-norm, or numpy.inf. The default
            is 2.
        k : list or numpy.ndarray, optional
            The multi-indices of which the norm is to be computed. The default
            is all the multi-indices of the object.

        Returns
        -------
        norm : numpy.ndarray
            The p-norm of the selected multi-indices.

        '''
        if k is None:
            k = np.arange(self.cardinal())
        if p == np.inf:
            norm = np.max(self.array[k, :], axis=1)
        else:
            norm = np.power(np.sum(self.array[k, :]**p, axis=1), 1/p)
        return norm

    def weighted_norm(self, p, w, k=None):
        '''
        Compute the weighted p-norm of multi-indices k in the object.

        Parameters
        ----------
        p : int or numpy.inf
            The positive real scalar p of the p-norm, or numpy.inf.
        w : list or numpy.ndarray
            The self.cardinal() weights used in the computation of the norm.
        k : list or numpy.ndarray, optional
            The multi-indices of which the norm is to be computed. The default
            is all the multi-indices of the object.

        Returns
        -------
        norm : numpy.ndarray
            The p-norm of the selected multi-indices.

        '''
        if k is None:
            k = np.arange(self.cardinal())
        if p == np.inf:
            norm = np.max(self.array[k, :]*np.tile(np.ravel(w),
                                                   (np.size(k), 1)), axis=1)
        else:
            norm = np.power(np.sum((self.array[k, :] *
                                    np.tile(np.ravel(w), (np.size(k), 1)))**p,
                                   axis=1), 1/p)
        return norm

    def sort_by_norm(self, p, mode='ascend'):
        '''
        Sort the multi-indices by increasing or decreasing p-norm.

        Parameters
        ----------
        p : int
            The positive real scalar p of the p-norm, or numpy.inf.
        mode : string, optional
            The sorting mode, equal to 'ascend' or 'descend'. The default is
            'ascend'.

        Returns
        -------
        tensap.MultiIndices
            The MultiIndices object with multi-indices sorted by increasing or
            decreasing p-norm.

        '''
        norm = self.norm(p)
        ind = np.argsort(norm)
        if mode == 'descend':
            ind = np.flip(ind)
        return MultiIndices(self.array[ind, :])

    def sort_by_weighted_norm(self, p, w, mode='ascend'):
        '''
        Sort the multi-indices by increasing or decreasing weighted p-norm.

        Parameters
        ----------
        p : int
            The positive real scalar p of the p-norm, or numpy.inf.
        w : list or numpy.ndarray
            The self.cardinal() weights used in the computation of the norm.
        mode : string, optional
            The sorting mode, equal to 'ascend' or 'descend'. The default is
            'ascend'.

        Returns
        -------
        tensap.MultiIndices
            The MultiIndices object with multi-indices sorted by increasing or
            decreasing weighted p-norm.

        '''
        norm = self.weighted_norm(p, w)
        ind = np.argsort(norm)
        if mode == 'descend':
            ind = np.flip(ind)
        return MultiIndices(self.array[ind, :])

    def sort(self, columns=None, mode='ascend'):
        '''
        Sort multi-indices column-wise using the column order provided in
        columns.

        The method first sorts according to the column columns[0], then sorts
        the equal coefficients of column columns[1] and so on.

        Parameters
        ----------
        columns : list or numpy.ndarray, optional
            The column order used to sort the multi-indices. The default is
            last to first column.
        mode : string, optional
            The sorting mode, equal to 'ascend' or 'descend'. The default is
            'ascend'.

        Returns
        -------
        tensap.MultiIndices
            The MultiIndices object with sorted multi-indices.

        '''
        if columns is None:
            columns = np.arange(self.array.shape[1]-1, -1, -1)

        array = np.array(self.array)
        for k in np.flip(columns):
            if k == columns[-1]:
                ind = np.argsort(array[:, k])
            else:
                ind = np.argsort(array[:, k], kind='mergesort')
            if mode == 'descend':
                ind = np.flip(ind)
            array = array[ind, :]
        return MultiIndices(array.astype(int))

    def add_indices(self, J):
        '''
        Return the union of multi-indices of self and J.

        Parameters
        ----------
        J : tensap.MultiIndices
            The second MultiIndices object.

        Returns
        -------
        tensap.MultiIndices
            The union of multi-indices of self and J.

        '''
        array = np.vstack((self.array, J.array))
        ind = np.unique(array, axis=0, return_index=True)[1]
        array = np.array([array[index, :] for index in sorted(ind)])
        return MultiIndices(array.astype(int))

    def remove_indices(self, J):
        '''
        Remove multi-indices J from self.

        Parameters
        ----------
        J : tensap.MultiIndices or int or list or numpy.ndarray
            The multi-indices to remove in a tensap.MultiIndices object,
            or their numbers as an int or in a list or numpy.ndarray.

        Returns
        -------
        tensap.MultiIndices
            The MultiIndices object with removed multi-indices in J.

        '''
        if isinstance(J, MultiIndices):
            ind = np.nonzero(np.all(self.array == J.array[:, np.newaxis],
                                    axis=2))[1]
        else:
            ind = J
        array = self.array[np.setdiff1d(range(self.cardinal()), ind), :]
        return MultiIndices(array.astype(int))

    def remove_dims(self, dims):
        '''
        Remove the dimensions in ind in the MultiIndices.

        Parameters
        ----------
        dims : int or list or numpy.ndarray
            The dimension(s) to be removed.

        Returns
        -------
        tensap.MultiIndices
            The MultiIndices object with removed dimenions in dims.

        '''
        array = self.array[:, np.setdiff1d(range(self.ndim()), dims)]
        ind = np.unique(array, axis=0, return_index=True)[1]
        array = np.array([array[index, :] for index in sorted(ind)])
        return MultiIndices(array.astype(int))

    def intersect_indices(self, J):
        '''
        Return the intersection of the multi-indices of self and J.

        Parameters
        ----------
        J : tensap.MultiIndices
            The second multi-indices.

        Returns
        -------
        tensap.MultiIndices
            The intersection of the multi-indices of self and J.
        ind_I : numpy.array
            The indices of the multi-indices in self common to self and J.
        ind_J : numpy.array
            The indices of the multi-indices in J common to self and J.

        '''
        ind_J, ind_I = np.nonzero(np.all(self.array == J.array[:, np.newaxis],
                                         axis=2))

        return MultiIndices(self.array[ind_I, :]), ind_I, ind_J

    def keep_indices(self, k):
        '''
        Keep the multi-indices k in self.

        Parameters
        ----------
        k : int or list or numpy.ndarray
            The number(s) of the multi-indices to keep.

        Returns
        -------
        tensap.MultiIndices
            The MultiIndices with retained indices.

        '''
        return MultiIndices(self.array[k, :])

    def keep_dims(self, dims):
        '''
        Keep the dimensions dims in self.

        Parameters
        ----------
        dims : int or list or numpy.ndarray
            The dimension(s) to keep.

        Returns
        -------
        tensap.MultiIndices
            The MultiIndices with retained dimensions.

        '''
        array = self.array[:, dims]
        ind = np.unique(array, axis=0, return_index=True)[1]
        array = np.array([array[index, :] for index in sorted(ind)])
        return MultiIndices(array.astype(int))

    def get_indices(self, k):
        '''
        Return the multi-indices k in self.

        Parameters
        ----------
        k : int or list or numpy.ndarray
            The numbers of the multi-indices to select.

        Returns
        -------
        numpy.ndarray
            The multi-indices k in self.

        '''
        return self.array[k, :]

    def is_downward_closed(self, m=0):
        '''
        Check whether or not the multi-index set is downward closed (or lower
        or monotone).

        Parameters
        ----------
        m : int, optional
            The lowest index for all dimensions. The default is 0.

        Returns
        -------
        cond : boolean
            Boolean indicating whether or not the multi-index set is downward
            closed.

        '''
        cond = True
        ind_test = np.arange(self.cardinal())
        while ind_test.size:
            p = self.array[ind_test[-1], :]
            Ip = MultiIndices.bounded_by(p, m)
            ind = np.all(Ip.array == self.array[:, np.newaxis], axis=2)
            ok = np.any(ind, axis=0)
            rep = np.nonzero(ind)[0]
            if not np.all(ok):
                cond = False
                return cond
            ind_test = np.setdiff1d(ind_test, rep)
        return cond

    def envelope(self, u):
        '''
        Compute the monotone envelope (or monotone majorant) of a bounded
        sequence u.

        Parameters
        ----------
        u : list or numpy.ndarray
            The bounded sequence.

        Returns
        -------
        env : numpy.ndarray
            The monotone envelope corresponding to the sequence defined by
            env_i = max_{j >= i} |u_j|

        '''
        array = self.array
        n = self.cardinal()
        assert np.size(u) == n, \
            ('The length of the sequence does not coincide with the number ' +
             'of multi-indices.')

        env = np.array(u)
        for i in range(n):
            ind_sup = np.all(array >= np.tile(array[i, :], (n, 1)), axis=1)
            env[i] = np.max(np.abs(u[ind_sup]))
        return env

    def get_maximal_indices(self):
        '''
        Return the set of maximal multi-indices contained in the downward
        closed multi-index set self.

        Returns
        -------
        tensap.MultiIndices
            The set of maximal multi-indices contained in the downward closed
            multi-index set self.

        '''
        dim = self.ndim()
        n = self.cardinal()
        neighbours = np.tile(np.transpose(np.expand_dims(self.array, 2),
                                          [0, 2, 1]), [1, dim, 1]) + \
            np.tile(np.transpose(np.expand_dims(np.eye(dim), 2),
                                 [2, 0, 1]), [n, 1, 1])
        neighbours = np.reshape(neighbours, [n*dim, dim],
                                order='F').astype(int)
        ok = np.any(np.all(neighbours == self.array[:, np.newaxis], axis=2),
                    axis=0)
        ok = np.reshape(ok, [n, dim], order='F')
        ind_max = self.array[np.logical_not(np.any(ok, axis=1)), :]
        return MultiIndices(ind_max.astype(int))

    def get_margin(self):
        '''
        Return the margin of the multi-index set self defined by the set of
        multi-indices i not in self such that it exists k in N^* s.t. i_k != 0
        implies i - e_k in self where e_k is the k-th Kronecker sequence.

        Returns
        -------
        tensap.MultiIndices
            The margin of self.

        '''
        dim = self.ndim()
        n = self.cardinal()
        neighbours = np.tile(np.transpose(np.expand_dims(self.array, 2),
                                          [0, 2, 1]), [1, dim, 1]) + \
            np.tile(np.transpose(np.expand_dims(np.eye(dim), 2),
                                 [2, 0, 1]), [n, 1, 1])
        neighbours = np.reshape(neighbours, [n*dim, dim],
                                order='F').astype(int)

        ind_marg = np.nonzero(np.all(neighbours == self.array[:, np.newaxis],
                                     axis=2))[1]
        ind_marg = neighbours[np.setdiff1d(range(neighbours.shape[0]),
                                           ind_marg), :]

        ind = np.unique(ind_marg, axis=0, return_index=True)[1]
        ind_marg = np.array([ind_marg[index, :] for index in sorted(ind)])

        return MultiIndices(ind_marg.astype(int))

    def get_reduced_margin(self):
        '''
        Return the reduced margin of the multi-index set self defined by the
        set of multi-indices i not in self such that for all k in N^* s.t.
        i_k != 0 implies i - e_k in self where e_k is the k-th Kronecker
        sequence.

        Returns
        -------
        tensap.MultiIndices
            The reduced margin of self.

        '''
        I_marg = self.get_margin()
        dim = self.ndim()
        neighbours = np.tile(np.transpose(np.expand_dims(I_marg.array, 2),
                                          [0, 2, 1]), [1, dim, 1]) - \
            np.tile(np.transpose(np.expand_dims(np.eye(dim), 2), [2, 0, 1]),
                    [I_marg.cardinal(), 1, 1])

        n = neighbours.shape[0]
        neighbours = np.reshape(neighbours, [n*dim, dim],
                                order='F').astype(int)

        ok = np.any(np.all(neighbours == self.array[:, np.newaxis], axis=2),
                    axis=0)
        is_out = np.any(neighbours < 0, axis=1)
        ok = np.logical_or(ok, is_out)
        ok = np.reshape(ok, [n, dim], order='F')
        keep = np.all(ok, axis=1)
        ind_marg_red = I_marg.array[keep, :]

        return MultiIndices(ind_marg_red.astype(int))

    def plot(self, *args):
        '''
        Plot the multi-index set self.

        See also the function plot_multi_indices.

        Parameters
        ----------
        *args : tuple
            Parameters used in tensap's function plot_multi_indices.

        Returns
        -------
        None.

        '''
        # TODO plot
        # tensap.plot_multi_indices(self, *args)

    @staticmethod
    def with_bounded_norm(d, p, m):
        '''
        Create the set of multi-indices in N^d with p-norm bounded by m, p>0.

        Parameters
        ----------
        d : int
            The dimension, a positive integer.
        p : int or numpy.inf
            The p of the p-norm, either a positive real scalar or numpy.inf.
        m : int
            The bound of the norm, a positive real scalar.

        Returns
        -------
        ind : tensap.MultiIndices
            The set of multi-indices in N^d with p-norm bounded by m.

        '''
        if p == np.inf:
            ind = MultiIndices.product_set(np.arange(m+1), d)
        elif p == 1:
            ind = MultiIndices(np.zeros(d, dtype=int))
            for i in range(m):
                ind = ind.add_indices(ind.get_margin())
        else:
            ind = MultiIndices(np.zeros(d, dtype=int))
            add = True
            while add:
                M = ind.get_margin()
                n = M.norm(p)
                k = np.nonzero(n <= m)[0]
                if np.all(np.logical_not(k)):
                    add = False
                else:
                    j = np.argsort(n[k])
                    M = M.keep_indices(k[j])
                    ind = ind.add_indices(M)
        return ind

    @staticmethod
    def with_bounded_weighted_norm(d, p, m, w):
        '''
        Create the set of multi-indices in N^d with weighted p-norm bounded by
        m, p>0.

        Parameters
        ----------
        d : int
            The dimension, a positive integer.
        p : int or numpy.inf
            The p of the p-norm, either a positive real scalar or numpy.inf.
        m : int
            The bound of the norm, a positive real scalar.
        w : list or numpy.ndarray
            The d weights defining the weighted p-norm of a multi-index i.
            |i|_{p, w} = (sum_{k=1}^d w_k^p i_k^p)^(1/p) for 0 < p < inf
            |i|_{Inf, w} = max_k w_k i_k

        Returns
        -------
        ind : tensap.MultiIndices
            The set of multi-indices in N^d with weighted p-norm bounded by m.

        '''
        ind = [np.arange(int(np.floor(m/x))+1) for x in w]
        ind = MultiIndices.product_set(ind)
        n = ind.weighted_norm(p, w)
        k = np.nonzero(n <= m)[0]
        j = np.argsort(n[k])
        return ind.keep_indices(k[j])

    @staticmethod
    def bounded_by(m, m0=0):
        '''
        Create the set of multi-indices bounded by m.

        Parameters
        ----------
        m : list or numpy.ndarray
            List or array of length d containing the highest indices in each
            dimension.
        m0 : int, optional
            The lowest index for all dimensions. The default is 0.

        Returns
        -------
        tensap.MultiIndices
            MultiIndices object containing the  product set
            (m0:m[0]) x ... x (m0:m[d-1]).

        '''
        return MultiIndices.product_set([np.arange(m0, x+1) for x in m])

    @staticmethod
    def product_set(L, d=None):
        '''
        Create the set of multi-indices obtained by a product of sets of
        indices.

        If ndim(L) is 1 and L contains a set of indices, the method returns
        a MultiIndices containing the product set L x ... x L (d times).

        If ndim(L) is 2 and L contains arrays of integers of size m_k,
        0 <= k <= d-1, the method returns a MultiIndices containing the product
        set L[0] x ... x L[d-1].

        Parameters
        ----------
        L : list or numpy.ndarray
            The grid or grids used to create the product set.
        d : int, optional
            The dimension of the set of multi-indices. The default is None,
            indicating to infer it from L.

        Raises
        ------
        ValueError
            If the provided arguments are wrong.

        Returns
        -------
        tensap.MultiIndices
            The set of multi-indices obtained by a product of sets of indices.

        '''
        if d is None:
            d = len(L)
        elif np.ndim(L) == 1:
            L = [L]*d
        else:
            raise ValueError('Wrong arguments.')

        L = [np.array(x) for x in L]
        N = [x.size for x in L]

        ind = list(np.unravel_index(range(np.prod(N)), N, order='F'))

        for i in range(d):
            ind[i] = L[i][ind[i]]

        return MultiIndices(np.transpose(np.array(ind)))

    @staticmethod
    def ind2sub(shape, ind):
        '''
        Create the set of multi-indices with array [I1, ..., Id] such that
        (I1, ..., Id) = np.unravel_index(np.ravel(ind), shape).

        Parameters
        ----------
        shape : list or numpy.ndarray
            The shape of the array.
        ind : list or numpy.ndarray
            The indices into the flattened version of an array of dimensions
            shape.

        Returns
        -------
        tensap.MultiIndices
            The MultiIndices created using the flat indices ind and shape.

        '''
        ind = np.unravel_index(np.ravel(ind), shape, order='F')
        return MultiIndices(np.transpose(ind))
