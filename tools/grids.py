'''
Module grids.

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

from abc import ABC, abstractmethod
import numpy as np
import tensap


class TensorGrid(ABC):
    '''
    Class TensorGrid: grids in product sets.

    Attributes
    ----------
    dim : int
        The dimension.
    grids: list
        List containing the grids.
    shape: numpy.ndarray
        The shapes of the grids.

    '''

    def __init__(self):
        '''
        Constructor for the class TensorGrid.

        Returns
        -------
        None.

        '''
        self.dim = None
        self.grids = None
        self.shape = None

    def ndim(self):
        '''
        Return the dimension of the underlying space.

        Returns
        -------
        int
            The dimension of the underlying space.

        '''
        return self.dim

    def shape(self, d=None):
        '''
        Return the shape of the grid, along the dimension d if provided.

        Equivalent to self.shape[d] if d is provided, self.shape otherwise.

        Parameters
        ----------
        d : int, optional
            The dimension for which the shape is asked. The default is None.

        Returns
        -------
        numpy.ndarray
            The shape of the grid.

        '''
        if d is None:
            return self.shape
        return self.shape[d]

    def plot_grid(self, *args, **kwargs):
        '''
        Plot the grid.

        Parameters
        ----------
        *args : tuple
            Parameters of the plot, see matplotlib.pyplot.plot for more
            information.

        Raises
        ------
        ValueError
            If the dimension is greater than 3.

        Returns
        -------
        None.

        '''
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        x = self.array()
        d = x.shape[1]
        fig = plt.figure()
        if d == 1:
            plt.plot(x, *args, **kwargs)
        elif d == 2:
            plt.plot(x[:, 0], x[:, 1], *args, **kwargs)
        elif d == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x[:, 0], x[:, 1], x[:, 2], *args, **kwargs)
        else:
            raise ValueError('The dimension must be less or equal than 3.')

    @abstractmethod
    def array(self):
        '''
        Return an array of shape (n, d) where n is the number of grid points
        and d is the dimension.

        Returns
        -------
        numpy.ndarray
            The TensorGrid as an array.

        '''

    @property
    @abstractmethod
    def size(self):
        '''
        Return the number of grid points.

        Returns
        -------
        int
            The number of grid points.

        '''


class FullTensorGrid(TensorGrid):
    '''
    Class FullTensorGrid: tensor product grid.

    '''

    def __init__(self, grids, dim=None):
        '''
        Constructor for the class FullTensorGrid.

        Parameters
        ----------
        grids : list or numpy.ndarray
            List containing arrays of shape (n_i, d_i), 1 <= i <= dim, or
            unidimensional list or numpy.ndarray replicated dim times
            (isotropic grid).
        dim : int, optional
            The dimension of the tensor grid. The default is None, indicating
            to infer it from grids.

        Returns
        -------
        None.

        '''

        tensap.TensorGrid.__init__(self)

        if dim is None:
            dim = len(grids)
        else:
            assert np.ndim(grids) == 1, \
                'The first argument must be a unidimensional grid.'
            grids = [np.reshape(grids, (-1, 1))]*dim
        grids = [np.atleast_2d(x) for x in grids]

        self.dim = dim
        self.shape = np.array([x.shape[0] for x in grids])
        self.grids = grids

    def array(self, ind=None):
        if ind is None:
            ind = self.multi_indices().array
        x = []
        for i in range(self.dim):
            x.append(self.grids[i][ind[:, i], :])
        return np.column_stack(x)

    def eval_at_indices(self, ind):
        return self.array(ind)

    @property
    def size(self):
        return np.prod(self.shape)

    def multi_indices(self):
        '''
        Return a set of multi-indices for indexing the grid.

        Returns
        -------
        tensap.MultiIndices
            Aset of multi-indices for indexing the grid.

        '''
        return tensap.MultiIndices.bounded_by(self.shape-1)

    def plot(self, y, *args):
        '''
        Plot the grid.

        Parameters
        ----------
        y : list or numpy.ndarray
            The values of the function on the grid.
        *args : tuple
            Parameters of the plot, see matplotlib.pyplot.plot for more
            information.

        Returns
        -------
        None.

        '''
        import matplotlib.pyplot as plt
        d = self.ndim()
        plt.figure()
        if d == 1:
            plt.plot(self.grids[0], y, *args)
        elif d == 2:
            from mpl_toolkits.mplot3d import Axes3D
            ax = plt.axes(projection='3d')
            x = self.array()
            ax.scatter(x[:, 0], x[:, 1], y, *args)
        else:
            raise ValueError('The dimension must be less or equal than 2.')

    @staticmethod
    def random(X, n):
        '''
        Generate a random FullTensorGrid from a RandomVector.

        Parameters
        ----------
        X : tensap.RandomVector
            The RandomVector used to generate the FullTensorGrid.
        n : list or numpy.ndarray
            Array of size d (or an integer for isotropic grid) containing
            the sizes of the grids in each dimension.

        Returns
        -------
        tensap.FullTensorGrid
            The random FullTensorGrid.

        '''
        d = X.size

        n = np.atleast_1d(n)
        if n.size == 1:
            n = np.tile(n, d)

        G = []
        for k in range(d):
            G.append(X.random_variables[k].random(n[k]))
        return FullTensorGrid(G)


class SparseTensorGrid(TensorGrid):
    '''
    Class SparseTensorGrid: sparse tensor product grid.

    '''

    def __init__(self, grids, indices, dim=None):
        '''
        Constructor for the class SparseTensorGrid

        Parameters
        ----------
        grids : list or numpy.ndarray
            List containing arrays of shape (n_i, d_i), 1 <= i <= dim, or
            unidimensional list or numpy.ndarray replicated dim times
            (isotropic grid).
        indices : tensap.MultiIndices
            The indices locating the non-zero coefficients (indices start at
            0).
        dim : int, optional
            The dimension of the tensor grid. The default is None, indicating
            to infer it from grids.

        Returns
        -------
        None.

        '''
        tensap.TensorGrid.__init__(self)

        T = tensap.FullTensorGrid(grids, dim)
        self.dim = T.dim
        self.grids = T.grids
        self.shape = T.shape
        self.indices = indices

    @property
    def size(self):
        return self.indices.cardinal()

    def array(self):
        ind = self.indices.to_list()

        x = list(self.grids)
        for k in range(len(x)):
            x[k] = x[k][ind[k], :]
        return np.hstack(x)
