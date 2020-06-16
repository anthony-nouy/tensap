'''
Initialization file for tensap (tensor approximation package).

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

from .linear_algebra.magic_indices import magic_indices

from .tensor_algebra.tensors.dimension_tree import DimensionTree
from .tensor_algebra.tensors.full_tensor import FullTensor
from .tensor_algebra.tensors.tree_based_tensor import TreeBasedTensor
from .tensor_algebra.tensors.canonical_tensor import CanonicalTensor
from .tensor_algebra.tensors.diagonal_tensor import DiagonalTensor
from .tensor_algebra.tensors.sparse_tensor import SparseTensor
from .tensor_algebra.tools.truncator import Truncator

from .tools.utils import *
from .tools.multi_indices import MultiIndices
from .tools.grids import TensorGrid, FullTensorGrid, SparseTensorGrid
from .tools.chebyshev_points import chebyshev_points

from .approximation.integration.integration_rules import *

from .functions.measures.copulas import *
from .functions.measures.measure import Measure
from .functions.measures.product_measure import ProductMeasure
from .functions.measures.probability_measure import ProbabilityMeasure
from .functions.measures.random_variable import RandomVariable
from .functions.measures.normal_random_variable import NormalRandomVariable
from .functions.measures.uniform_random_variable import UniformRandomVariable
from .functions.measures.empirical_random_variable import \
    EmpiricalRandomVariable
from .functions.measures.discrete_random_variable import DiscreteRandomVariable
from .functions.measures.discrete_measure import DiscreteMeasure
from .functions.measures.random_vector import RandomVector
from .functions.measures.random_multi_indices import random_multi_indices
from .functions.polynomials.polynomials import *
from .functions.polynomials.orthonormal_polynomials import *
from .functions.sets.is_in import is_in
from .functions.function import Function
from .functions.functional_tensor import FunctionalTensor
from .functions.user_defined_function import UserDefinedFunction
from .functions.tensorizer import Tensorizer
from .functions.tensorized_function import TensorizedFunction
from .functions.compositional_model_function import \
    CompositionalModelFunction
from .functions.multivariate_functions_benchmark import \
    multivariate_functions_benchmark

from .approximation.tools.model_selection import ModelSelection
from .approximation.bases.functional_basis_array import FunctionalBasisArray
from .approximation.bases.functional_bases import FunctionalBases
from .approximation.bases.functional_basis import FunctionalBasis
from .approximation.bases.polynomial_functional_basis import \
    PolynomialFunctionalBasis
from .approximation.bases.sub_functional_basis import SubFunctionalBasis
from .approximation.bases.user_defined_functional_basis import \
    UserDefinedFunctionalBasis
from .approximation.bases.full_tensor_product_functional_basis import \
        FullTensorProductFunctionalBasis
from .approximation.bases.sparse_tensor_product_functional_basis import \
        SparseTensorProductFunctionalBasis

from .approximation.learning.loss_functions import *
from .approximation.learning.learning import Learning
from .approximation.learning.linear_model_learning import LinearModelLearning
from .approximation.learning.linear_model_learning_custom_loss import \
    LinearModelLearningCustomLoss
from .approximation.learning.linear_model_learning_square_loss import \
    LinearModelLearningSquareLoss
from .approximation.learning.linear_model_learning_density_l2 import \
    LinearModelLearningDensityL2
from .approximation.tensor_approximation.tensor_learning.tensor_learning \
    import TensorLearning
from .approximation.tensor_approximation.tensor_learning.\
    tree_based_tensor_learning import TreeBasedTensorLearning
from .approximation.tensor_approximation.tensor_learning.\
    canonical_tensor_learning import CanonicalTensorLearning
from .approximation.tensor_approximation.principal_component_analysis.\
    tensor_principal_component_analysis import TensorPrincipalComponentAnalysis
from .approximation.tensor_approximation.principal_component_analysis.\
    functional_tensor_principal_component_analysis import \
        FunctionalTensorPrincipalComponentAnalysis

# Disable tensorflow's warning messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

__pdoc__ = {"tutorials": False}
