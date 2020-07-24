'''
Module model_selection.

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
import tensap


class ModelSelection:
    '''
    Class ModelSelection.

    Attributes
    ----------
    pen_shape : function
        Function specifying the penalization shape.
    data : dict
        Dictionnary containing the data used for the model selection.
        data['complexity'] contains the complexities of the models, and
        data['empirical_risk'] the associated empirical risk values.
    gap_factor : int or float
        Multiplicative factor used in the slope heuristic.

    '''

    def __init__(self):
        '''
        Constructor for the class ModelSelection.

        Returns
        -------
        None.

        '''
        self.data = {'complexity': [],
                     'empirical_risk': []}
        self.pen_shape = lambda x: x
        self.gap_factor = 2

    def m_lambda(self, lbda):
        '''
        Compute the argument of the minimum of the penalized risk for given
        values of the penalization factor lbda.

        Parameters
        ----------
        lbda : list or numpy.ndarray
            The values of the penalization factor.

        Returns
        -------
        numpy.ndarray
            The argument of the minimum of the penalized risk for the values of
            the penalization factor.

        '''
        comp = np.array(self.data['complexity'])
        risk = np.array(self.data['empirical_risk'])
        return np.array([np.argmin(risk + l * self.pen_shape(comp)) for
                         l in np.atleast_1d(lbda)])

    def lambda_path(self):
        '''
        Return the path of possible values of lambda and associated arguments
        of the minimum of the penalized risk.


        Returns
        -------
        numpy.ndarray
            The path of possible values of lambda.
        numpy.ndarray
            The path of the arguments of the minimum of the penalized risk
            associated with the path of possible values of lambda.

        '''
        comp = np.array(self.data['complexity'])
        risk = np.array(self.data['empirical_risk'])

        path = []
        m_path = []
        lambda_0 = 0
        lambda_current = 0
        ind = np.argmin(risk)
        ok = True

        while ok:
            with np.errstate(all='ignore'):
                lambda_current = (risk - risk[ind]) / \
                    (self.pen_shape(comp[ind]) - self.pen_shape(comp))
                lambda_current[lambda_current <= lambda_0] = np.nan
                lambda_current[lambda_current == np.inf] = np.nan

            ok = not np.all(np.isnan(lambda_current))
            if ok:
                lambda_0 = np.nanmin(lambda_current)
                if not np.isnan(lambda_0):
                    ind = np.nanargmin(lambda_current)
                    path.append(lambda_0)
                    m_path.append(ind)

        if path == []:
            path = 0
            m_path = 0
        else:
            path = np.atleast_1d(path)
            m_path = np.atleast_1d(m_path)
            ind_min = np.argmin(risk)
            path = np.concatenate(([path[0]/2], path, [path[-1]*2]))
            m_path = np.concatenate(([ind_min], m_path, [m_path[-1]]))
        return np.array(path), np.array(m_path)

    def slope_heuristic(self, lambda_path=None, m_path=None):
        '''
        Apply the slope heuristic to the path of possible values of lambda to
        compute its optimal value lambda_hat and associated argument of the
        minimum of the penalized risk m_hat.

        Parameters
        ----------
        lambda_path : list or numpy.ndarray, optional
            The path of possible values of lambda. The default is None,
            indicating to compute it using the method lambda_path.
        m_path : list or numpy.ndarray, optional
            The path of the arguments of the minimum of the penalized risk
            associated with the path of possible values of lambda. The default
            is None, indicating to compute it using the method lambda_path.

        Returns
        -------
        lambda_hat : float
            The value of lambda determined using the slope heuristic.
        m_hat : int
            The model number associated with the value of lambda determined
            using the slope heuristic.
        lambda_path : numpy.ndarray
            The path of possible values of lambda.
        m_path : numpy.ndarray
            The path of the arguments of the minimum of the penalized risk
            associated with the path of possible values of lambda.

        '''
        comp = np.array(self.data['complexity'])

        # If all the complexities are equal, choose the first model
        if np.all(np.diff(comp) == 0):
            lambda_hat = 0
            m_hat = 0
            lambda_path = lambda_hat
            m_path = m_hat
            return lambda_hat, m_hat, lambda_path, m_path

        if lambda_path is None and m_path is None:
            lambda_path, m_path = self.lambda_path()
        l_mid = (lambda_path[:-1] + lambda_path[1:]) / 2
        gaps = comp[self.m_lambda(l_mid[:-1])] - comp[self.m_lambda(l_mid[1:])]
        ind = np.argmax(gaps)
        lambda_hat = self.gap_factor * lambda_path[ind+1]
        m_hat = self.m_lambda(lambda_hat)
        return lambda_hat, m_hat, lambda_path, m_path

    @staticmethod
    def complexity(x, *args, **kwargs):
        '''
        Return the complexity associated to the input argument's type.

        See also complexity_functional_basis_array,
        complexity_tree_based_tensor.

        Parameters
        ----------
        x : list or numpy.ndarray or tensap.FunctionalBasisArray or
        tensap.FunctionalTensor or tensap.TreeBasedTensor
            The object(s) of which the complexity is computed.
        *args, **kwargs : tuples
            Additional parameters for the methods
            complexity_functional_basis_array and complexity_tree_based_tensor.

        Raises
        ------
        ValueError
            If the argument x is not of correct type.

        Returns
        -------
        float or list
            The complexity(ies) associated with the object(s).

        '''
        if isinstance(x, (list, np.ndarray)):
            return [ModelSelection.complexity(y, *args, **kwargs) for y in x]
        if isinstance(x, tensap.FunctionalBasisArray):
            return np.size(x.data)
        if isinstance(x, tensap.FunctionalTensor):
            return ModelSelection.complexity(x.tensor, *args, **kwargs)
        if isinstance(x, tensap.TreeBasedTensor):
            return ModelSelection.complexity_tree_based_tensor(x, *args,
                                                               **kwargs)
        raise ValueError('Wrong argument.')

    @staticmethod
    def complexity_functional_basis_array(x, fun=None):
        '''
        Return the complexity associated with a FunctionalBasisArray.

        Parameters
        ----------
        x : tensap.FunctionalBasisArray
            The FunctionalBasisArray of which the complexity is computed.
        fun : str, optional
            Name of the function applied to the array to extract the storage
            complexity. The default is 'storage'.

        Returns
        -------
        float
            The complexity associated with the FunctionalBasisArray.

        '''
        if fun is None:
            fun = 'storage'
        return eval('x.' + fun + '()')

    @staticmethod
    def complexity_tree_based_tensor(x, fun=None, c_type='standard'):
        '''
        Return the complexity associated with the TreeBasedTensor.

        Parameters
        ----------
        x : tensap.TreeBasedTensor
            The TreeBasedTensor of which the complexity is computed.
        fun : str, optional
            Name of the function applied to the array to extract the storage
            complexity. The default is 'storage'. Can also be 'sparse_storage'
            or 'sparse_leaves_storage' for instance.
        c_type : str, optional
            The complexity type. The default is 'standard'. Can also be
            'stiefel' or 'grassman'.

        Raises
        ------
        ValueError
            If the complexity type is neither 'standard' nor 'stiefel' nor
            'grassman'.

        Returns
        -------
        float or list
            The complexity(ies) associated with the TreeBasedTensor(s).

        '''
        if fun is None:
            fun = 'storage'

        if isinstance(x, (list, np.ndarray)):
            return ModelSelection.complexity(x, fun, c_type)
        if isinstance(x, tensap.FunctionalTensor):
            return ModelSelection.complexity(x.tensor, fun, c_type)
        assert isinstance(x, tensap.TreeBasedTensor), \
            'The first argument should be a TreeBasedTensor.'

        if c_type == 'standard':
            comp = eval('x.' + fun + '()')
        elif c_type == 'grassman':
            comp = eval('x.' + fun + '()') - np.sum(x.ranks**2)
        elif c_type == 'stiefel':
            comp = eval('x.' + fun + '()') - np.sum(x.ranks*(x.ranks+1)/2)
        else:
            raise ValueError('Wrong argument.')
        return comp
