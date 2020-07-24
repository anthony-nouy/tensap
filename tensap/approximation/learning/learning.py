'''
Module learning.

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

import tensap


class Learning:

    def __init__(self, loss):
        assert isinstance(loss, tensap.LossFunction), \
            'Must provide a tensap.LossFunction object.'

        self.loss_function = loss
        self.display = True
        self.model_selection = True
        self.model_selection_options = {'stop_if_error_increase': False}
        self.error_estimation = False
        self.error_estimation_type = 'leave_out'
        self.error_estimation_options = {'correction': True}
        self.training_data = None
        self.test_error = False
        self.test_data = None

    @staticmethod
    def linear_model(loss_function):
        if isinstance(loss_function, tensap.SquareLossFunction):
            model = tensap.LinearModelLearningSquareLoss()
        elif isinstance(loss_function, tensap.DensityL2LossFunction):
            model = tensap.LinearModelLearningDensityL2()
        elif isinstance(loss_function, tensap.CustomLossFunction):
            model = tensap.LinearModelLearningCustomLoss(loss_function)
        return model

    def risk_estimation(self, fun, sample, *args):
        return self.loss_function.risk_estimation(fun, sample, *args)
