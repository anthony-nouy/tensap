'''
Module loss_functions.

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


class LossFunction(ABC):
    '''
    Class LossFunction.

    '''

    def risk_estimation(self, fun, sample, *args, nargout=1):
        '''
        Compute an estimation of the risk associated with the function fun and
        the loss function using the provided sample.

        Parameters
        ----------
        fun : function
            The function to which the risk is associated.
        sample : list
            The sample used to estimate the risk.
        *args : tuple
            Additional parameters used by a specific loss function.
        nargout : int, optional
            Specifies the number of output variables. The default is 1,
            returning only the risk. Set to 2 to also return the so-called
            estimated reference risk.

        Returns
        -------
        np.float
            The estimated risk.
        np.float
            The estimated reference risk.

        '''
        loss, loss_ref = self.eval(fun, sample, *args, nargout=2)
        risk = np.mean(loss)
        if nargout == 1:
            return risk
        risk_ref = np.mean(loss_ref)
        return risk, risk_ref

    @abstractmethod
    def eval(self, fun, sample, *args, nargout):
        '''
        Evaluate the loss function using the function fun and the provided
        sample.

        Parameters
        ----------
         fun : function
            The function used to evaluate the loss function
        sample : list
            The sample used to evaluate the loss function
        *args : tuple
            Additional parameters used by a specific loss function.
        nargout : int, optional
            Specifies the number of output variables. The default is 1,
            returning only the evaluations of the loss. Set to 2 to also return
            the so-called reference evaluations of the loss.

        Returns
        -------
        None.

        '''

    @abstractmethod
    def relative_test_error(self, fun, sample, *args):
        '''
        Compute the relative test error associated with the function fun and
        the loss function using the provided sample.

        Parameters
        ----------
        fun : function
            The function to which the error is associated.
        sample : list
            The sample used to estimate the risk.
        *args : tuple
            Additional parameters used by a specific loss function.

        Returns
        -------
        np.float
            The relative test error.

        '''

    @abstractmethod
    def test_error(self, fun, sample, *args):
        '''
        Compute the test error associated with the function fun and
        the loss function using the provided sample.

        Parameters
        ----------
        fun : function
            The function to which the error is associated.
        sample : list
            The sample used to estimate the risk.
        *args : tuple
            Additional parameters used by a specific loss function.

        Returns
        -------
        np.float
            The relative test error.

        '''


class SquareLossFunction(LossFunction):
    '''
    Class SquareLossFunction.

    Attributes
    ----------
    error_type : string, optional
        The error type. The default is 'relative'. Can also be 'absolute'.

    '''

    def __init__(self):
        '''
        Constructor for the class SquareLossFunction.

        Returns
        -------
        None.

        '''
        self.error_type = 'relative'

    def eval(self, fun, sample, *args, nargout=1):
        try:
            y_pred = fun.eval(sample[0])
        except Exception:
            try:
                y_pred = fun(sample[0])
            except Exception:
                y_pred = fun

        y_pred = np.atleast_2d(y_pred)
        y_true = np.atleast_2d(sample[1])
        loss = np.sum((y_pred - y_true)**2, 1)

        if nargout == 1:
            return loss
        loss_ref = np.sum(y_true**2, 1)
        return loss, loss_ref

    def relative_test_error(self, fun, sample, *args):
        risk, risk_ref = self.risk_estimation(fun, sample, *args, nargout=2)
        return np.sqrt(risk / risk_ref)

    def test_error(self, fun, sample, *args):
        if self.error_type == 'absolute':
            error = np.sqrt(self.risk_estimation(fun, sample, *args))
        elif self.error_type == 'relative':
            error = self.relative_test_error(fun, sample, *args)
        else:
            raise ValueError('The error_type property must be set to "risk" ' +
                             'or "relative".')
        return error


class DensityL2LossFunction(LossFunction):
    '''
    Class DensityL2LossFunction.

    Attributes
    ----------
    error_type : string, optional
        The error type. The default is 'relative'.

    '''

    def __init__(self):
        '''
        Constructor for the class DensityL2LossFunction.

        Returns
        -------
        None.

        '''
        self.error_type = 'risk'

    def eval(self, fun, sample, *args, nargout=1):
        try:
            y_pred = fun.eval(sample[0])
        except Exception:
            try:
                y_pred = fun(sample[0])
            except Exception:
                y_pred = fun

        try:
            l_ref = fun.norm()**2
        except Exception:
            try:
                l_ref = args[0]**2
            except Exception:
                print('Input fun must have a method norm, or its norm ' +
                      'must be provided in last argument.')
                l_ref = np.nan

        loss = l_ref - 2*y_pred
        if nargout == 1:
            return loss
        return loss, l_ref

    def relative_test_error(self, fun, sample, *args):
        raise NotImplementedError('Relative test error not available for ' +
                                  'DensityL2LossFunction.')

    def test_error(self, fun, sample, *args):
        assert self.error_type == 'risk', \
            'The error_type attribute must be set to "risk".'
        return self.risk_estimation(fun, sample, *args)


class CustomLossFunction(LossFunction):
    '''
    Class CustomLossFunction.

    Attributes
    ----------
    loss_function : tensorflow.function
        The custom loss function, defined using tensorflow operations.
    error_function : function, optional
        The function used to compute the test error. The default is None,
        indicating to use the risk function.
    relative_error_function : function, optional
        The function used to compute the relative test error. The default is
        None, indicating to use the risk function.
    error_type : string, optional
        The error type. The default is 'relative'. Can also be 'absolute'.

    '''

    def __init__(self, loss_function):
        self.loss_function = loss_function
        self.error_function = None
        self.relative_error_function = None
        self.error_type = 'absolute'

    def risk_estimation(self, fun, sample, *args, nargout=1):
        try:
            from tensorflow import reduce_mean
        except ImportError:
            reduce_mean = np.mean
        loss, loss_ref = self.eval(fun, sample, *args, nargout=2)
        risk = reduce_mean(loss)
        if nargout == 1:
            return risk
        risk_ref = reduce_mean(loss_ref)
        return risk, risk_ref

    def eval(self, fun, sample, *args, nargout=1):
        try:
            y_pred = fun.eval(sample[0])
        except Exception:
            try:
                y_pred = fun(sample[0])
            except Exception:
                y_pred = fun

        try:
            y_true = sample[1]
        except Exception:
            y_true = None

        loss = self.loss_function(y_true, y_pred)

        if nargout == 1:
            return loss
        loss_ref = self.loss_function(y_true, 0*y_pred)
        return loss, loss_ref

    def relative_test_error(self, fun, sample, *args):
        assert self.relative_error_function is not None, \
            ('Relative test error not available, please provide '
             'self.relative_error_function.')
        return self.relative_error_function(fun, sample)

    def test_error(self, fun, sample, *args):
        if self.error_function is None:
            error = self.risk_estimation(fun, sample, *args)
        else:
            error = self.error_function(fun, sample, *args)
        return error
