'''
Tutorial on digits recognition using tree-based tensor formats.

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

from sklearn import datasets, metrics
import random
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import tensap

# %% Data import and preparation
DIGITS = datasets.load_digits()
DATA = DIGITS.images.reshape((len(DIGITS.images), -1))
DATA = DATA / np.max(DATA)  # Scaling of the data

# %% Patch reshape of the data: the patches are consecutive entries of the data
PS = [4, 4]  # Patch size
DATA = np.array([np.concatenate(
    [np.ravel(np.reshape(DATA[k, :], [8]*2)[PS[0]*i:PS[0]*i+PS[0],
                                            PS[1]*j:PS[1]*j+PS[1]]) for
     i in range(int(8/PS[0])) for j in range(int(8/PS[1]))]) for
    k in range(DATA.shape[0])])
DIM = int(int(DATA.shape[1]/np.prod(PS)))

# %% Probability measure
print('Dimension %i' % DIM)
X = tensap.RandomVector(tensap.DiscreteRandomVariable(np.unique(DATA)), DIM)

# %% Training and test samples
P_TRAIN = 0.9  # Proportion of the sample used for the training

N = DATA.shape[0]
TRAIN = random.sample(range(N), int(np.round(P_TRAIN*N)))
TEST = np.setdiff1d(range(N), TRAIN)
X_TRAIN = DATA[TRAIN, :]
X_TEST = DATA[TEST, :]
Y_TRAIN = DIGITS.target[TRAIN]
Y_TEST = DIGITS.target[TEST]

# One hot encoding (vector-valued function)
Y_TRAIN = tf.one_hot(Y_TRAIN.astype(int), 10, dtype=tf.float64)
Y_TEST = tf.one_hot(Y_TEST.astype(int), 10, dtype=tf.float64)

# %% Approximation bases: 1, cos and sin for each pixel of the patch
FUN = [lambda x: np.ones((np.shape(x)[0], 1))]
for i in range(np.prod(PS)):
    FUN.append(lambda x, j=i: np.cos(np.pi / 2*x[:, j]))
    FUN.append(lambda x, j=i: np.sin(np.pi / 2*x[:, j]))

BASES = [tensap.UserDefinedFunctionalBasis(FUN, X.random_variables[0],
                                           np.prod(PS)) for _ in range(DIM)]
BASES = tensap.FunctionalBases(BASES)

# %% Loss function: cross-entropy custom loss function
LOSS = tensap.CustomLossFunction(
        lambda y_true, y_pred: tf.nn.sigmoid_cross_entropy_with_logits(
            logits=y_pred, labels=y_true))


def error_function(y_pred, sample):
    '''
    Return the error associated with a set of predictions using a sample, equal
    to the number of misclassifications divided by the number of samples.

    Parameters
    ----------
    y_pred : numpy.ndarray
        The predictions.
    sample : list
        The sample used to compute the error. sample[0] contains the inputs,
        and sample[1] the corresponding outputs.

    Returns
    -------
    int
        The error.

    '''
    try:
        y_pred = y_pred(sample[0])
    except Exception:
        pass
    return np.count_nonzero(np.argmax(y_pred, 1) - np.argmax(sample[1], 1)) / \
        sample[1].numpy().shape[0]


LOSS.error_function = error_function

# %% Learning in tree-based tensor format
TREE = tensap.DimensionTree.balanced(DIM)
IS_ACTIVE_NODE = np.full(TREE.nb_nodes, True)
SOLVER = tensap.TreeBasedTensorLearning(TREE, IS_ACTIVE_NODE, LOSS)

SOLVER.tolerance['on_stagnation'] = 1e-10
SOLVER.initialization_type = 'random'
SOLVER.bases = BASES
SOLVER.training_data = [X_TRAIN, Y_TRAIN]
SOLVER.test_error = True
SOLVER.test_data = [X_TEST, Y_TEST]

SOLVER.rank_adaptation = True
SOLVER.rank_adaptation_options['max_iterations'] = 15
SOLVER.model_selection = True
SOLVER.display = True

SOLVER.alternating_minimization_parameters['display'] = False
SOLVER.alternating_minimization_parameters['max_iterations'] = 10
SOLVER.alternating_minimization_parameters['stagnation'] = 1e-10

# Options dedicated to the LinearModelCustomLoss object
SOLVER.linear_model_learning.options['max_iterations'] = 10
SOLVER.linear_model_learning.options['stagnation'] = 1e-10
SOLVER.linear_model_learning.optimizer.learning_rate = 1e3

SOLVER.rank_adaptation_options['early_stopping'] = True
SOLVER.rank_adaptation_options['early_stopping_factor'] = 10

T0 = time.time()
F, OUTPUT = SOLVER.solve()
T1 = time.time()
print(T1-T0)

# %% Display of the results
F_X_TEST = np.argmax(F(X_TEST), 1)
Y_TEST_NP = np.argmax(Y_TEST.numpy(), 1)

print('\nAccuracy = %2.5e\n' % (1 - np.count_nonzero(F_X_TEST - Y_TEST_NP) /
                                Y_TEST_NP.shape[0]))

IMAGES_AND_PREDICTIONS = list(zip(DIGITS.images[TEST], F_X_TEST))
for i in np.arange(1, 19):
    plt.subplot(3, 6, i)
    plt.imshow(IMAGES_AND_PREDICTIONS[i][0],
               cmap=plt.cm.gray_r, interpolation='nearest')
    plt.axis('off')
    plt.title('Pred.: %i' % IMAGES_AND_PREDICTIONS[i][1])

print('Classification report:\n%s\n'
      % (metrics.classification_report(Y_TEST_NP, F_X_TEST)))
MATRIX = metrics.confusion_matrix(Y_TEST_NP, F_X_TEST)
plt.matshow(MATRIX)
plt.title('Confusion Matrix')
plt.show()
print('Confusion matrix:\n%s' % MATRIX)
