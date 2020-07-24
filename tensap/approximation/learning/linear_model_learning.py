'''
Module linear_model_learning.

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


class LinearModelLearning(tensap.Learning):

    def __init__(self, loss):
        super().__init__(loss)
        self.basis = None
        self.basis_eval = None
        self.basis_eval_test = None
        self.regularization = False
        self.regularization_type = 'l1'
        self.regularization_options = {'alpha': 0}
        self.basis_adaptation = False
        self.basis_adaptation_path = None
        self.options = {}

    def initialize(self):
        # If the test error cannot be computed, it is disabled
        if self.test_error and self.basis is None and \
                self.basis_eval_test is None:
            print('The test error cannot be computed.')
            self.test_error = False

        # Bases evaluation
        try:
            if self.training_data is not None and self.basis_eval is None:
                if isinstance(self.training_data, list) and \
                        self.training_data[0] is not None:
                    self.basis_eval = self.basis.eval(self.training_data[0])
                elif not isinstance(self.training_data, list) and \
                        self.training_data is not None:
                    self.basis_eval = self.basis.eval(self.training_data)
                else:
                    raise ValueError('Must provide input training data.')

            if self.test_error and self.test_data is not None and \
                    self.basis_eval_test is None:
                if isinstance(self.test_data, list) and \
                        self.test_data[0] is not None:
                    self.basis_eval_test = self.basis.eval(
                        self.test_data[0])
                elif not isinstance(self.test_data, list) and \
                        self.test_data is not None:
                    self.basis_eval_test = self.basis.eval(self.test_data)
                else:
                    raise ValueError('Must provide input test data.')
        except Exception:
            pass
