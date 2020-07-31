============
Introduction
============

tensap (Tensor Approximation Package) is a Python package for the
approximation of functions and tensors, available on GitHub at
https://github.com/anthony-nouy/tensap, or through its GitHub page
https://anthony-nouy.github.io/tensap/.

To install from PyPi, run::

    pip install tensap

Alternatively, you can install tensap directly from github by running::

    pip install git+git://github.com/anthony-nouy/tensap@master

The package tensap features low-rank tensors (including canonical,
tensor-train and tree-based tensor formats or tree tensor networks),
sparse tensors, polynomials, and allows the plug-in of other
approximation tools. It provides different approximation methods based
on interpolation, least-squares projection or statistical learning.

The package is shipped with tutorials showing its main applications. A
documentation is also available.

At minimum, tensap requires the packages numpy and scipy. The packages
tensorflow and sklearn are required for some applications.

**FullTensor**
==============

A **FullTensor** **X** represents an order :math:`d` tensor
:math:`X \in {\mathbb{R}}^{N_1 \times \cdots \times N_d}`, or
multidimensional array of size :math:`N_1\times \ldots \times N_d`.
The entries of :math:`X` are :math:`X_{i_1, \ldots, i_d}`, with
:math:`(i_1,\ldots,i_d)` a tuple of indices, where
:math:`i_\nu \in \{0,\ldots,N_\nu-1\}` is related to the
:math:`\nu`-th mode of the tensor.

We present in this section how to create a **FullTensor** using tensap,
and several possible operations with such an object. For an introduction
to tensor calculus, we refer to the monograph [hackbusch2019tensor]_.

For examples of use, see the tutorial file
``tutorials\tensor_algebra\tutorial_FullTensor.py``.

Creating a **FullTensor**
-------------------------

| Provided with an array **data** of shape **[N\_1, ..., N\_d]**, the
  command **X = tensap.FullTensor(data)** returns a tensor
  :math:`X \in {\mathbb{R}}^{N_1 \times \cdots \times N_d}`, with order
  **X.order = d** and shape **X.shape = (N\_1, ..., N\_d)**. The number
  of entries of **X** is given by
  :math:`{\texttt{\detokenize{X.size = X.storage()}}} = \prod_{i=1}^d N_i`.
  The number of nonzero entries of **X** is given by
  **X.sparse\_storage()**.

It is also possible to generate a **FullTensor** with entries:

-  equal to 0 with **tensap.FullTensor.zeros([N\_1, ..., N\_d])**,

-  equal to 1 with **tensap.FullTensor.ones([N\_1, ..., N\_d])**,

-  drawn randomly according to the uniform distribution on
   :math:`[0, 1]` with **tensap.FullTensor.rand([N\_1, ..., N\_d])**,

-  drawn randomly according to the standard gaussian distribution with
   **tensap.FullTensor.randn([N\_1, ..., N\_d])**,

-  different from 0 only on the diagonal, provided in **diag_data**,
   with **tensap.FullTensor.diag(diag_data, d)** (generating a tensor
   of order d and shape :math:`[N, \ldots, N]`, with :math:`N = len(diag_data)`,

-  generated using a provided **generator** with
   **tensap.FullTensor.create(generator, [N\_1, ..., N\_d])**.

Accessing the entries of a **FullTensor** 
------------------------------------------

The entries of a tensor **X** can be accessed with the method
**eval\_at\_indices**: **X.eval\_at\_indices(ind)** returns the entries
of :math:`X` indexed by the list **ind** containing the indices to
access in each dimension.

Extracting diagonal entries.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a tensor :math:`X \in {\mathbb{R}}^{N, \ldots, N}`, the command
**X.eval\_diag()** returns the diagonal entries
:math:`X_{i, \ldots, i}`, :math:`i = 1, \ldots, N`, of the tensor. The
command **X.eval\_diag(dims)** returns the entries
:math:`X_{i, \ldots, i}`, with :math:`i` in **ind**.

Extracting a sub-tensor.
^^^^^^^^^^^^^^^^^^^^^^^^

A sub-tensor can be extracted from **X** with the method
**sub\_tensor**: for an order-3 **FullTensor** **X** of size
:math:`N_1\times N_2 \times N_3`, **X.sub\_tensor([0, 1], ’:’, 2)**
returns a sub-tensor of size :math:`2\times N_2 \times 1` containing the
entries :math:`X_{i_1,i_2,i_3}` with :math:`i_1\in \{0,1\}`,
:math:`0\le i_2 \le N_2-1` and :math:`i_3=2`.

Permuting the modes of a **FullTensor**
---------------------------------------

The methods **transpose** and **itranspose** permute the dimensions of a
tensor **X**, given a permutation **dims** of :math:`\{1, \ldots, d\}`.
They are such that **X = X.transpose(dims).itranspose(dims)**.

Reshaping a **FullTensor**.
---------------------------

The command **X.reshape(shape)** reshapes a **FullTensor** using a
column-major order (e.g. used in Fortran, Matlab, R). It relies on the
numpy’s reshape function with Fortran-like index (argument
**order=’F’**). For a tuple :math:`(i_1,\ldots,i_d)`, we define

.. math:: \overline{i_1,\ldots,i_d} = i_1 + N_1(i_{2-1}-1) + N_1N_2(i_{3-1}-1) + \ldots + N_1\ldots N_{d-1}(i_d-1) .

A tensor :math:`X` is be identified with a vector
:math:`\mathrm{vec}(X)` whose entries are
:math:`\mathrm{vec}(X)_{\overline{i_1,\ldots,i_d}}`. This vector can be
obtained with the command **X.reshape(N)** with
**N=numpy.prod(X.shape)**.

:math:`\alpha`-Matricization.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For :math:`\alpha \subset \{1,\ldots,d\}` an its complementary subset
:math:`\alpha^c` in :math:`\{1,\ldots,d\}`, an
:math:`\alpha`-matricization of a tensor :math:`X` is a matrix :math:`M`
of size
:math:`(\prod_{i \in \alpha} N_i) \times (\prod_{i \in \alpha^c} N_i)`,
such that
:math:`X_{i_1,\ldots,i_d} = M_{\overline{i_\alpha},\overline{i_{\alpha^c}}}`
with :math:`i_\alpha = (i_\nu)_{\nu\in \alpha}`. It can be obtained with
**X.matricize(alpha)**, which returns a **FullTensor** or order
:math:`2`. The matricization relies on the method **reshape**.

Orthogonalization.
^^^^^^^^^^^^^^^^^^

It is possible to obtain a representation of a tensor :math:`X` such
that its :math:`\alpha`-matricization is an orthogonal matrix (i.e. with
orthogonal columns) using the method **X.orth(alpha)**.

Norms and singular-values
-------------------------

Computing the Frobenius norm of a **FullTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The command **X.norm()** returns the Frobenius norm
:math:`\Vert X \Vert_F` of :math:`X`, defined by

.. math:: \Vert X \Vert_F^2 = \sum_{i_1}^{N_1} \cdots \sum_{i_d}^{N_d} X_{i_1, \ldots, i_d}^2.

Computing the :math:`\alpha`-singular values and :math:`\alpha`-principal components of a **FullTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a subset :math:`\alpha \subset \{1, \ldots, d\}` and its
complementary subset :math:`\alpha^c`, the :math:`\alpha`-matricization
:math:`M` of :math:`X` admits a singular value decomposition

.. math:: M_{i_\alpha,i_{\alpha^c}} = \sum_{k} \sigma^k v^k_{i_\alpha} w^k_{i_\alpha^c}

where the :math:`\sigma^k` are the singular values of :math:`M` and the
:math:`v^k` the corresponding left singular vectors, or principal
components of :math:`M`. They are respectively called the
:math:`\alpha`-singular values and :math:`\alpha`-principal components
of :math:`X`. The :math:`\alpha`-singular values are obtained with
**X.singular\_values()**. The :math:`\alpha`-principal components (and
:math:`\alpha`-singular values) are obtained with
**X.alpha\_principal\_components(alpha)**, which is equivalent to
**X.matricize(alpha).principal\_components()**.

Operations with **FullTensor**
------------------------------

Outer product.
^^^^^^^^^^^^^^

| The outer product :math:`X \circ Y` of two tensors
  :math:`X\in {\mathbb{R}}^{N_1 \times \cdots \times N_d}` and
  :math:`Y \in {\mathbb{R}}^{\hat N_1 \times \cdots \times \hat N_{\hat d}}`
  is a tensor
  :math:`Z \in {\mathbb{R}}^{N_1 \times \ldots \times N_d \times \hat N_1 \times \cdots \times \hat N_{\hat d}}`
  of order :math:`d + \hat d` with entries

  .. math:: {Z}_{i_1,\ldots,i_d,j_1,\ldots,j_{\hat d}} =  X_{i_1, \ldots,i_ d} Y_{j_1, \ldots, j_{\hat d}}

  It is provided by **X.tensordot(Y, 0)**, similarly to numpy’s
  tensordot function.

Kronecker product.
^^^^^^^^^^^^^^^^^^

The Kronecker product :math:`X\otimes Y` of two tensors :math:`X` and
:math:`Y` of the same order :math:`d=\hat d` is a tensor :math:`Z` of
size :math:`N_1\hat N_1 \times  \ldots \times N_d \hat N_{\hat d}` with
entries

.. math:: Z_{\overline{i_1j_1},\ldots,\overline{i_dj_d}} = X_{i_1,\ldots,i_d} Y_{j_1,\ldots,j_d}.

It is given by the command **kron**, which is similar to numpy’s kron
function, but for arbitrary tensors.

Hadamard product.
^^^^^^^^^^^^^^^^^

The Hadamard (elementwise) product :math:`X \circledast Y` of two
tensors :math:`X` and :math:`Y` of the same order and size is obtained
through the command **\_\_mul\_\_(X,Y)**, which returns a tensor
:math:`Z` with entries

.. math:: Z_{i_1,\ldots,i_d} = X_{i_1,\ldots,i_d} Y_{i_1,\ldots,i_d}

Contracted product.
^^^^^^^^^^^^^^^^^^^

| For :math:`I\subset \{1,\ldots,d\}` and
  :math:`J \subset \{1,\ldots,\hat d\}` with :math:`\#I = \#J`, **Z =
  X.tensordot(Y, I, J)** performs the mode :math:`(I,J)`-contracted
  product of :math:`X` and :math:`Z` which is a tensor Z of order
  :math:`d + \hat d - \#I - \#J` with entries

  .. math::

     {Z}_{(i_\nu)_{\nu \notin I}, (j_\mu)_{\mu \notin J}} =
                 \sum_{\substack{i_\nu=1 \\ \nu \in I}}^{N_\nu} \sum_{\substack{j_\mu=1 \\ \mu \in J}}^{N_\mu} 
                 \prod_{\nu \in I} \prod_{\mu \in J} 
                 \delta_{i_\nu, j_\mu} X_{i_1, \ldots,i_ d} Y_{j_1, \ldots, j_{\hat d}}

  with :math:`\delta_{i,j}` the Kronecker delta, that is a contraction
  of tensors :math:`X` and :math:`Y` along dimensions :math:`I` of
  :math:`X` and :math:`J` of :math:`Y`. For example, for
  order-\ :math:`4` tensors :math:`X` and :math:`Y`, **Z =
  X.tensordot(Y, [0,1], [1,2])** returns a tensor :math:`Z` or order
  :math:`4` such that

  .. math:: Z_{i_3,i_4,j_1,j_4} = \sum_{i_1,i_2} X_{i_1,i_2,i_3,i_4} Y_{j_1,i_1,i_2,j_4} .

  The method **tensordot\_eval\_diag** provides the diagonal (or
  entries with equal pairs of indices) of the result of the method
  **tensor\_dot**, but at a cost lower than when using **X.tensordot(Y,
  I, J).eval\_diag()**.

| For example, for order-\ :math:`4` tensors :math:`X` and :math:`Y`,
  **X.tensordot\_eval\_diag(Y,[0,1],[1,2],[2,3],[0,3])** returns the
  diagonal of :math:`Z`, i.e. an order-one tensor :math:`M` with entries

  .. math:: M_k = Z_{k,k,k,k} = \sum_{i_1,i_2} X_{i_1,i_2,k,k} Y_{k,i_1,i_2,k}

  **X.tensordot\_eval\_diag(Y,[0,1],[1,2],[2,3],[0,3],diag = True)**
  returns a tensor :math:`M` of order :math:`2` with entries

  .. math:: M_{k_1,k_2} = Z_{k_1,k_2,k_1,k_2} = \sum_{i_1,i_2} X_{i_1,i_2,k_1,k_3} Y_{k,i_1,i_2,k}

  **X.tensordot\_eval\_diag(Y,[0,1],[1,2],[2],[0])** returns the
  diagonal of :math:`Z`, i.e. a tensor :math:`M` of order :math:`3`
  :math:`v` with entries

  .. math:: M_{k,i_4,j_4} = Z_{k,i_4,k,j_4} = \sum_{i_1,i_2} X_{i_1,i_2,k,i_4} Y_{k,i_1,i_2,j_4}

Dot product.
^^^^^^^^^^^^

The dot product of two tensors :math:`X` and :math:`Y` with same shape
:math:`[N_1, \ldots, N_d]`, defined by

.. math:: ( X, Y ) = \sum_{\substack{i_\nu = 1 \\ \nu = 1, \ldots, d}}^{N_\nu} X_{i_1, \ldots, i_d} Y_{i_1, \ldots, i_d},

can be obtained with **X.dot(Y)**. It is equivalent to **X.tensordot(Y,
range(X.order), range(Y.order))**.

Contractions with matrices or vectors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given a tensor :math:`X` and a list of matrices
:math:`M = [M^1, ..., M^d]`, the command **Z =
X.tensor\_matrix\_product(M)** returns an order-d tensor :math:`Z` whose
entries are

.. math:: Z_{i_1, \ldots, i_d} = \sum_{\substack{k_\nu = 1 \\ \nu = 1, \ldots, d}}^{N_\nu} X_{k_1, \ldots, k_d} \prod_{\nu = 1}^d M^\nu_{i_\nu, k_\nu}

The same method exists for vectors instead of matrices:
**tensor\_vector\_product**. Similarly to **tensordot\_eval\_diag**, the
method **tensor\_matrix\_product\_eval\_diag** evaluates the diagonal of
the result of **tensor\_matrix\_product**, with a lower cost.

Tensor formats
==============

Here we present tensor formats available in tensap, which are structured
formats of tensors in :math:`\mathbb{R}^{N_1\times \ldots \times N_d}.`
For a detailed description of methods, see the description of the
corresponding methods for **FullTensor** in . For an introduction to
tensor formats, we refer to the monograph [hackbusch2019tensor]_ and the survey
[nouy:2017_morbook]_.

**CanonicalTensor**
-------------------

The entries of an order-\ :math:`d` tensor
:math:`X \in {\mathbb{R}}^{N_1 \times \cdots \times N_d}` in canonical
format can be written

.. math::

   \label{eq:CanonicalTensor}
               X_{i_1, \ldots, i_d} = \sum_{k=1}^r C_k U^1_{i_1, k} \cdots U^d_{i_d, k},

with :math:`r` the canonical rank, and where the
:math:`U_\nu = (U^\nu_{i_\nu, k})_{1\le i_\nu \le N_\nu , 1\le k \le r}`
are order-two tensors.

Creating a **CanonicalTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To create a canonical tensor in tensap, one can use the command
**tensap.CanonicalTensor(C, U)**, where **C** contains the
:math:`(C_k)_{k=1}^d`, and **U** is a list containing the :math:`U^\nu`,
:math:`1\le \nu\le d`.

| The storage complexity of such a tensor, obtained with
  **X.storage()**, is equal to :math:`r(1 + N_1 + \cdots + N_d)`.
| It is also possible to generate a **CanonicalTensor** with entries

-  equal to 0 with **tensap.CanonicalTensor.zeros(r, [N\_1, ...,
   N\_d])**,

-  equal to 1 with **tensap.CanonicalTensor.ones(r, [N\_1, ...,
   N\_d])**,

-  drawn randomly according to the uniform distribution on
   :math:`[0, 1]` with **tensap.CanonicalTensor.rand(r, [N\_1, ...,
   N\_d])**,

-  drawn randomly according to the standard gaussian distribution with
   **tensap.CanonicalTensor.randn(r, [N\_1, ..., N\_d])**,

-  generated using a provided **generator** with
   **tensap.CanonicalTensor.create** **(generator, r, [N\_1, ...,
   N\_d])**.

Converting a **CanonicalTensor** to a **FullTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **CanonicalTensor** **X** can be converted to a **FullTensor**
(introduced in Section [sec:FullTensor]) with the command **X.full()**.

Converting a **CanonicalTensor** to a **TreeBasedTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **CanonicalTensor** **X** can be converted to a **TreeBasedTensor**
(introduced in Section [sec:TreeBasedTensor]) with the command
**X.tree\_based\_tensor(tree, is\_active\_node)**, with **tree** a
**DimensionTree** object, and **is\_active\_node** a list or array of
booleans indicating if each node of the tree is active.

Accessing the diagonal of a **CanonicalTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a canonical tensor
:math:`X \in {\mathbb{R}}^{N \times \ldots \times N}`, the command
**X.eval\_diag()** returns the diagonal :math:`X_{i, \ldots, i}`,
:math:`i = 1, \ldots, N`, of the tensor. The method **eval\_diag** can
also be used to evaluate the diagonal in a subset of dimensions **dims**
of the tensor with **X.eval\_diag(dims)**, which returns a
**CanonicalTensor**.

Computing the Frobenius norm of a **CanonicalTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The command **X.norm()** returns the Frobenius norm of :math:`X`. The
Frobenius norm of :math:`X` is equal to the Frobenius norm of its core
:math:`C` if **X.is\_orth** is **True**.

Computing the derivative of **CanonicalTensor** with respect to one of its parameters.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given an order-\ :math:`d` canonical tensor :math:`X` in
:math:`{\mathbb{R}}^{N \times \cdots \times N}`, the command
**X.parameter\_gradient\_eval\_diag(k)**, for :math:`1 \leq k \leq d`,
returns the derivative

.. math:: \left.\frac{\partial X_{i_1, \ldots, i_d}}{\partial U^k}\right|_{i_1=\cdots=i_d=i}, \; i = 1, \ldots, N.

The derivative of :math:`X` with respect to its core :math:`C`, that
writes

.. math:: \left.\frac{\partial X_{i_1, \ldots, i_d}}{\partial C}\right|_{i_1=\cdots=i_d=i}, \; i = 1, \ldots, N,

is obtained with **X.parameter\_gradient\_eval\_diag(d+1)**.

The method **parameter\_gradient\_eval\_diag** is used in the
statistical learning algorithms presented in Section
[sec:TensorLearning].

Performing operations with **CanonicalTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some operations between tensors are implemented for **DiagonalTensor**
(see for a detailed description of the operations): the Kronecker
product with **kron**, the contraction with matrices with
**tensor\_matrix\_product**, the evaluation of the diagonal of a
contraction with matrices with **tensor\_matrix\_product\_eval\_diag**,
the dot product with **dot**.

Given a tensor :math:`X` and a list of matrices
:math:`M = [M^1, ..., M^d]`, the command **Z =
X.tensor\_matrix\_product(M)** returns an order-d tensor :math:`Z` whose
entries are

.. math:: Z_{i_1, \ldots, i_d} = \sum_{k=1}^r \sum_{\substack{k_\nu = 1 \\ \nu = 1, \ldots, d}}^{N_\nu} C_k U^1_{k_1, k} \cdots U^d_{k_d, k} \prod_{\nu = 1}^d M^\nu_{i_\nu, k_\nu}

The method **tensor\_matrix\_product\_eval\_diag** evaluates the
diagonal of the result of **tensor\_matrix\_product**.

The dot product of two canonical tensors :math:`X` and :math:`Y` with
same shape :math:`[N_1, \ldots, N_d]` can be obtained with **X.dot(Y)**.

**DiagonalTensor**
------------------

A diagonal tensor
:math:`X \in {\mathbb{R}}^{N_1 \times \cdots \times N_d}` is a tensor
whose entries :math:`X_{i_1, \ldots, i_d}` are non-zero only if
:math:`i_1 = \cdots = i_d`.

Creating a **DiagonalTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To create a diagonal tensor in tensap, one can use the command
**tensap.DiagonalTensor(D, d)**, where **D** (of length :math:`r`)
contains the diagonal of the tensor, and **d** is the order of the
tensor. The result if an order :math:`d` tensor in
:math:`\mathbb{R}^{r\times \ldots \times r} = \mathbb{R}^{ r^d}`.

The sparse storage complexity of such a tensor, obtained with
**X.sparse\_storage()**, is equal to **r = len(D)**. Its storage
complexity, not taking into account the fact that only the diagonal is
non-zero, is equal to :math:`r^d` and obtained with **X.storage()**.

It is also possible to generate a **DiagonalTensor** with entries

-  equal to 0 with **tensap.DiagonalTensor.zeros(r, d)**,

-  equal to 1 with **tensap.DiagonalTensor.ones(r, d)**,

-  drawn randomly according to the uniform distribution on
   :math:`[0, 1]` with **tensap.DiagonalTensor.rand(r, d)**,

-  drawn randomly according to the standard gaussian distribution with
   **tensap.DiagonalTensor.randn(r, d)**,

-  generated using a provided **generator** with
   **tensap.DiagonalTensor.create** **(generator, r, d)**.

Converting a **DiagonalTensor** to a **FullTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **DiagonalTensor** **X** can be converted to a **FullTensor**
(introduced in Section [sec:FullTensor]) with the command **X.full()**.

Converting a **DiagonalTensor** to a **SparseTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **DiagonalTensor** **X** can be converted to a **SparseTensor**
(introduced in Section [sec:SparseTensor]) with the command
**X.sparse()**.

Converting a **DiagonalTensor** to a **TreeBasedTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **DiagonalTensor** **X** can be converted to a **TreeBasedTensor**
(introduced in Section [sec:TreeBasedTensor]) with the command
**X.tree\_based\_tensor(tree, is\_active\_node)**, with **tree** a
**DimensionTree** object, and **is\_active\_node** a list or array of
booleans indicating if each node of the tree is active.

Accessing the entries of a **DiagonalTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| The entries of the tensor **X** can be accessed with the method
  **eval\_at\_indices**: **X.eval\_at\_indices(ind)** returns the
  entries of :math:`X` indexed by the list **ind** containing the
  indices to access in each dimension.
| A sub-tensor can be extracted from **X** with the method
  **sub\_tensor**.

For a tensor :math:`X \in {\mathbb{R}}^{N \times \ldots \times N}`, the
command **X.eval\_diag()** returns the diagonal
:math:`X_{i, \ldots, i}`, :math:`i = 1, \ldots, N`, of the tensor. The
method **eval\_diag** can also be used to evaluate the diagonal in some
dimensions **dims** of the tensor with **X.eval\_diag(dims)**.

Computing the Frobenius norm of a **DiagonalTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The command **X.norm()** returns the Frobenius norm of :math:`X`.

Performing operations with **DiagonalTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some operations between tensors are implemented for **DiagonalTensor**
(see for a detailed description of the operations): the outer product
with **tensordot**, the evaluation of the diagonal (or subtensors) of an
outer product with **tensordot\_eval\_diag**, the Kronecker product with
**kron**, the contraction with matrices or vectors with
**tensor\_matrix\_product** or **tensor\_vector\_product** respectively,
the evaluation of the diagonal of a contraction with matrices with
**tensor\_matrix\_product\_eval\_diag**, the dot product with **dot**.

**SparseTensor**
----------------

A sparse tensor
:math:`X \in {\mathbb{R}}^{N_1 \times \cdots \times N_d}` is a tensor
whose entries :math:`X_{i_1, \ldots, i_d}` are non-zero only for
:math:`(i_1, \ldots, i_d) \in I`, with :math:`I` a set of multi-indices.

Creating a **SparseTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To create a sparse tensor :math:`{\texttt{\detokenize{X}}}` in tensap,
one can use the command **tensap.SparseTensor(D, I, [N\_1, ...,
N\_d])**, where **D** contains the non-zero entries of :math:`X`, **I**
is a **tensap.MultiIndices** containing the indices of its non-zero
enties, and where :math:`N_1, \ldots, N_d` is its shape.

The sparse storage complexity of such a tensor, obtained with
**X.sparse\_storage()**, is equal to :math:`\text{card}(I)`. Its storage
complexity, not taking into account the sparsity, is equal to
:math:`N_1 \cdots N_d` and can be accessed with **X.storage()**.

Converting a **SparseTensor** to a **FullTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **SparseTensor** **X** can be converted to a **FullTensor**
(introduced in Section [sec:FullTensor]) with the command **X.full()**.

Converting a **FullTensor** to a **SparseTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **FullTensor** **X** can be converted to a **SparseTensor**
(introduced in Section [sec:SparseTensor]) with the command
**X.sparse()**.

Accessing the entries of a **SparseTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The entries of the tensor **X** can be accessed with the method
**eval\_at\_indices**: **X.eval\_at\_indices(ind)** returns the entries
of :math:`X` indexed by the list **ind** containing the indices to
access in each dimension.

A sub-tensor can be extracted from **X** with the method
**sub\_tensor**.

For a tensor :math:`X \in {\mathbb{R}}^{N, \ldots, N}`, the command
**X.eval\_diag()** returns the diagonal :math:`X_{i, \ldots, i}`,
:math:`i = 1, \ldots, N`, of the tensor. The method **eval\_diag** can
also be used to evaluate the diagonal in some dimensions **dims** of the
tensor with **X.eval\_diag(dims)**.

Reshaping a **SparseTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The method **reshape** reshapes a **SparseTensor** using the
Fortran-like index order of numpy’s reshape function.

The methods **transpose** and **itranspose** permute the dimensions of a
tensor **X**, given a permutation **dims** of :math:`\{1, \ldots, d\}`.
They are such that **X = X.transpose(dims).itranspose(dims)**.

Computing the Frobenius norm of a **SparseTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The command **X.norm()** returns the Frobenius norm of :math:`X`.

Performing operations with **SparseTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some operations between tensors are implemented for **SparseTensor**
(see for a detailed description of the operations): the Kronecker
product with **kron**, the contraction with matrices or vectors with
**tensor\_matrix\_product** or **tensor\_vector\_product** respectively,
the evaluation of the diagonal of a contraction with matrices with
**tensor\_matrix\_product\_eval\_diag**, the dot product with **dot**.

**TreeBasedTensor** and **DimensionTree**
-----------------------------------------

We present in this section the **DimensionTree** and **TreeBasedTensor**
objects. For examples of use, see the tutorial file
``tutorials\tensor_algebra\tutorial_DimensionTree.py`` and
``tutorials\tensor_algebra\tutorial_TreeBasedTensor.py``.

**DimensionTree**
^^^^^^^^^^^^^^^^^

| A dimension tree :math:`T` is a collection of non-empty subsets of
  :math:`D = \{1, \ldots, d\}` which is such that (i) all nodes
  :math:`\alpha \in T` are non-empty subsets of :math:`D`, (ii)
  :math:`D` is the root of :math:`T`, (iii) every node
  :math:`\alpha \in T` with :math:`\#\alpha \ge 2` has at least two
  children and the set of children of :math:`\alpha`, denoted by
  :math:`S(\alpha)`, is a non-trivial partition of :math:`\alpha`, and
  (iv) every node :math:`\alpha` with :math:`\#\alpha = 1` has no child
  and is called a leaf (see for example Figure [fig:treeExamples]).
| We let
  :math:`\operatorname{depth}(T) = \max_{\alpha \in T} \operatorname{level}(\alpha)`
  be the depth of :math:`T`, and :math:`{\mathcal{L}}(T)` be the set of
  leaves of :math:`T`, which are such that :math:`S(\alpha) = \emptyset`
  for all :math:`\alpha \in {\mathcal{L}}(T)`.

.32

=[circle,fill=black] child node [active,label=below::math:`\{1\}`] child
node [active,label=below::math:`\{2\}`] child node
[active,label=below::math:`\{3\}`] child node
[active,label=below::math:`\{4\}`] ;

.32

=[circle,fill=black] =[sibling distance=15mm] =[sibling distance=15mm]
=[sibling distance=15mm] child node [active,label=below::math:`\{1\}`]
child node [active,label=above right:\ :math:`\{2,3,4\}`] child node
[active,label=below::math:`\{2\}`] child node [active,label=above
right:\ :math:`\{3,4\}`] child node [active,label=below::math:`\{3\}`]
child node [active,label=below::math:`\{4\}`] ;

.32

=[circle,fill=black] =[sibling distance=20mm] =[sibling distance=10mm]
child node [active,label=above left:\ :math:`\{1,2\}`] child node
[active,label=below::math:`\{1\}`] child node
[active,label=below::math:`\{2\}`] child node [active,label=above
right:\ :math:`\{3,4\}`] child node [active,label=below::math:`\{3\}`]
child node [active,label=below::math:`\{4\}`] ;

Creating a **DimensionTree**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **DimensionTree** is characterized by its adjacency matrix and the
dimension associated with each leaf node: **T =
tensap.DimensionTree(dims, adjacency\_matrix)**. The adjacency matrix of
a dimension tree :math:`T` can be accessed with **T.adjacency\_matrix**.
The dimension associated with each leaf node can be accessed with
**T.dim2ind**.

Denoting by **order** the number of leaf nodes, it is possible to create

-  a trivial tree with **tensap.DimensionTree.trivial(order)** (Figure
   [fig:trivialTree]),

-  a linear tree with **tensap.DimensionTree.linear(order)** (Figure
   [fig:linearTree]),

-  a balanced tree with
   **tensap.DimensionTree.balanced(order)**\ (Figure
   [fig:balancedTree]),

-  a random tree with **tensap.DimensionTree.random(order, arity)**,
   with **arity** the arity of the tree, equal to the maximum number of
   children per node (randomly selected in an interval if provided).

Finally, a dimension tree can be created by extracting a sub-tree from
an existing tree :math:`T` with **T.sub\_dimension\_tree(root)** where
**root** is the node in :math:`T` that will become the root node of the
extracted tree.

Displaying a **DimensionTree**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **DimensionTree** can be displayed with the command **T.plot()**. The
dimension associated with each leaf node can be plotted on the tree with
**T.plot\_dims()**. Finally, the tree can be plotted with some quantity
displayed at each node with **T.plot\_with\_labels\_at\_nodes(labels)**.

Accessing properties of the tree.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The number of nodes of a dimension tree :math:`T` is given by
**T.nb\_nodes**.

The parent of :math:`\alpha`, denoted by :math:`P(\alpha)`, can be
obtained with **T.parent(alpha)**, and its ascendants :math:`A(\alpha)`
and descendants :math:`D(\alpha)` by **T.ascendants(alpha)** and
**T.descendants(alpha)**, respectively. The children of :math:`\alpha`
are given by **T.children(alpha)**. The command
**T.child\_number(alpha)** returns :math:`i^\gamma_\alpha`, for
:math:`\alpha \in T \setminus \{D\}` and :math:`\gamma = P(\alpha)`,
which is such that :math:`\alpha` is the :math:`i^\gamma_\alpha`-th
child of :math:`\gamma`. For instance, in the tree of Figure
[fig:linearTree], the node :math:`\alpha = \{3,4\}` is the second child
of :math:`\gamma = \{2,3,4\}`.

The level of a node :math:`\alpha` is denoted by
:math:`\operatorname{level}(\alpha)`. The levels are defined such that
:math:`\operatorname{level}(D) = 0` and
:math:`\operatorname{level}(\beta) = \operatorname{level}(\alpha) + 1`
for :math:`\beta \in S(\alpha)`. The nodes of :math:`T` with level
:math:`l` are returned by **T.nodes\_with\_level(l)**.

The leaf nodes :math:`\alpha \in {\mathcal{L}}(T)` are such that
**T.is\_leaf[alpha-1]** is **True**.

**TreeBasedTensor**
~~~~~~~~~~~~~~~~~~~

| Given a dimension tree :math:`T`, a **TreeBasedTensor** **X** is a
  tensor in *tree-based format* (see
  [Falco2018SEMA]_, [hackbusch2019tensor]_). It represents
  an order :math:`d` tensor
  :math:`X \in {\mathbb{R}}^{N_1 \times \cdots \times N_d}` in the set
  of tensors with :math:`\alpha`-ranks bounded by some integer
  :math:`r_\alpha`, :math:`\alpha \in T`. Such a tensor admits a
  representation

  .. math:: X_{i_1, \ldots, i_d} = \sum_{\substack{1 \leq k_\beta \leq r_\beta \beta \in T\setminus \{D\}}} \prod_{\alpha \in T\setminus {\mathcal{L}}(T)} C^\alpha_{(k_\beta)_{\beta \in S(\alpha)},k_\alpha} \prod_{\alpha \in {\mathcal{L}}(T)} C^{\alpha}_{i_\alpha,k_\alpha},

  with :math:`C^\alpha`, :math:`\alpha \in T`, some tensors that
  parameterize the representation of :math:`X`. When :math:`T` is a
  binary tree, the corresponding format is the so-called hierarchical
  Tucker (HT) format. The particular case of a linear binary tree is the
  tensor train Tucker format.

| The *Tucker format* corresponds to a trivial tree
  :math:`T=\{\{1\},\ldots,\{d\},\{1,\ldots,d\}\}` and admits the
  representation

  .. math:: X_{i_1, \ldots, i_d} = \sum_{k_1=1}^{r_1} \ldots \sum_{k_d=1}^{r_d} C^{1,\ldots,d}_{k_1,\ldots,k_d} C^{1}_{i_1,k_1} \ldots C^{d}_{i_d,k_d}.

A *degenerate tree-based format* is defined as the set of tensors with
:math:`\alpha`-ranks bounded by some integer :math:`r_\alpha`, for all
:math:`\alpha` in a subset :math:`A` of :math:`T`. The set :math:`A`
corresponds to active nodes, which should contain all interior nodes
:math:`T\setminus{\mathcal{L}}(T)`. A **TreeBasedTensor** **X** with
active nodes :math:`A` admits a representation.

.. math:: X_{i_1, \ldots, i_d} = \sum_{\substack{1 \leq k_\beta \leq r_\beta \beta \in A \setminus \{D\}}} \prod_{\alpha \in A\setminus {\mathcal{L}}(T)} C^\alpha_{(k_\beta)_{\beta \in S(\alpha)},k_\alpha} \prod_{\alpha \in {\mathcal{L}}(T) \cap A} C^{\alpha}_{i_\alpha,k_\alpha},

with :math:`C^\alpha`, :math:`\alpha \in A`, some tensors that
parameterize the representation of :math:`X`.

| The *tensor train format* is a degenerate tree-based format with a
  linear tree :math:`T` and all leaf nodes inactive except the first
  one, that means :math:`A = \{\{1\},\{1,2\}, \ldots, \{1,\ldots,d\}\}`.
  A tensor :math:`X` in tensor train format admits a representation

  .. math:: X_{i_1,\ldots,i_d} = \sum_{k_1=1}^{r_1} \ldots \sum_{k_{d-1}=1}^{r_{d-1}} C^1_{1,i_1,k_1} C^2_{k_1,i_2,k_2} \ldots C^{d-1}_{k_{d-2},i_{d-1},k_{d-1}} C^d_{k_{d-1},i_d,1}

  with tensor :math:`C^\nu` and rank :math:`r_\nu` associated with the
  node :math:`\alpha = \{1,\ldots,\nu\}`.

| For a more detailed presentation of tree-based formats (possibly
  degenerate) and more examples, see [nouy:2017hopca]_.
| If the rank :math:`r_D` associated with the root node is different
  from :math:`1`, a **TreeBasedTensor** **X** represents a tensor of
  order :math:`d+1` with entries :math:`X_{i_1,\ldots,i_d,k_D}`,
  :math:`1\le k_D \le r_D`. I can be used to defined vector-valued
  functional tensors (see ).

Creating a **TreeBasedTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| A **TreeBasedTensor** is created with the command **X =
  tensap.TreeBasedTensor(C, T)**, with **C** the list of **FullTensor**
  objects representing the :math:`C^\alpha`, :math:`\alpha \in T`, and
  **T** a **DimensionTree**. If some entries of the list **C**
  corresponding to leaf nodes are empty, it creates a degenerate tensor
  format, with :math:`T\setminus A` corresponding to the empty entries
  of **C**.
| It is possible to create a **TreeBasedTensor** in tensor-train format
  with the command **tensap.TreeBasedTensor.tensor\_train(C)**, with
  **C** a list containing the tensors :math:`C^1,\ldots,C^d`.
| Given a **DimensionTree** **T**, it is also possible to generate a
  **TreeBasedTensor** with entries

-  equal to 0 with **tensap.TreeBasedTensor.zeros(T, r, s, I)**,

-  equal to 1 with **tensap.TreeBasedTensor.ones(T, r, s, I)**,

-  drawn randomly according to the uniform distribution on
   :math:`[0, 1]` with **tensap.TreeBasedTensor.rand(T, r, s, I)**,

-  drawn randomly according to the standard gaussian distribution with
   **tensap.TreeBasedTensor.randn(T, r, s, I)**,

-  generated using a provided **generator** with
   **tensap.TreeBasedTensor.create** **(generator, T, r, s, I)**,

where **r** is a list containing the :math:`\alpha`-ranks,
:math:`\alpha \in T`, or **’random’**, **s** is a list containing the
sizes :math:`N_1, \ldots, N_d`, or **’random’**, and **I** is a list of
booleans indicating if the node :math:`\alpha` is active,
:math:`\alpha \in T`, or **’random’**.

Storage complexity. 
^^^^^^^^^^^^^^^^^^^^

| The storage complexity of **X** is given by
  :math:`{\texttt{\detokenize{X.size = X.storage()}}}` and returns the
  number of entries in tensors :math:`C^\alpha`, :math:`\alpha\in A`.
| The storage complexity of **X** taking into account the sparsity in
  the :math:`C^\alpha`, :math:`\alpha \in T`, is given by
  **X.sparse\_storage()**. It returns the number of non-zero entries in
  tensors :math:`C^\alpha`, :math:`\alpha\in A`.
| The storage complexity of **X** taking into account the sparsity only
  in the leaf nodes is given by **X.sparse\_leaves\_storage()**.

Displaying a **TreeBasedTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A graphical representation of a **TreeBasedTensor** **X** can be
obtained with the command **X.plot()**. Labels can be added to the nodes
of the tree, as well as a title, with **X.plot(labels, title)**.

Converting a **TreeBasedTensor** to a **FullTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **TreeBasedTensor** **X** can be converted to a **FullTensor**
(introduced in Section [sec:FullTensor]) with the command **X.full()**.

Converting a **FullTensor** to a **TreeBasedTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **FullTensor** **X** can be converted to a **TreeBasedTensor** with
the command **X.tree\_based\_tensor()**. The associated dimension tree
is a trivial tree with active nodes.

Accessing the entries of a **TreeBasedTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The entries of the tensor **X** can be accessed with the method
**eval\_at\_indices**: **X.eval\_at\_indices(ind)** returns the entries
of :math:`X` indexed by the list **ind** containing the indices to
access in each dimension.

A sub-tensor can be extracted from **X** with the method **sub\_tensor**
(see )

For a tensor :math:`X \in {\mathbb{R}}^{N, \ldots, N}`, the command
**X.eval\_diag()** returns the diagonal :math:`X_{i, \ldots, i}`,
:math:`i = 1, \ldots, N`, of the tensor. The method **eval\_diag** can
also be used to evaluate the diagonal in some dimensions **dims** of the
tensor with **X.eval\_diag(dims)**.

Obtaining an orthonormal representation of a **TreeBasedTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The command **X.orth()** returns a representation of **X** where all the
core tensors except the root core represent orthonormal bases of
principal subspaces.

The command **X.orth\_at\_node(alpha)** returns a representation of
**X** where all the core tensors except the one of node :math:`\alpha`
represent orthonormal bases of principal subspaces. The core tensor
:math:`C^\alpha` of the node :math:`\alpha` is such that the tensor
writes

.. math:: X_{i_\alpha, i_{\alpha^c}} = \sum_{k} \sum_{l} C^\alpha_{k,l} u_l(i_\alpha) w_k(i_{\alpha^c}),

where the :math:`u_l` are orthonormal tensors and the :math:`w_k` are
orthonormal tensors. This orthonormality of the representation can be
checked by computing the Gram matrices of the bases of minimal subspaces
associated with the nodes of the tree with **X.gramians()**.

Modifying the tree structure of a **TreeBasedTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is possible to modify the tree of a **TreeBasedTensor** **X** by
permuting two of its nodes :math:`\alpha` and :math:`\beta` given a
relative tolerance **tol** with **X.permute\_nodes([alpha, beta],
tol)**.

The leaves of the tree can also be permuted with the command
**X.permute\_leaves(perm, tol)**, where **perm** is a permutation of
:math:`(1, \ldots, d)`.

The method **optimize\_dimension\_tree** tries random permutations of
nodes to minimize the storage complexity of a tree-based tensor
:math:`X`: **X.optimize\_dimension\_tree(tol, n)** tries :math:`n`
random permutations and returns a **TreeBasedTensor** **Y** which is
such that **Y.storage()** is less or equal than **X.storage()**. The
nodes to permute are drawn according to probability measures favoring
high decreases of the ranks while maintaining a permutation cost as low
as possible (see [grelier2019learning]_).

The similar method **optimize\_leaves\_permutations** focuses on the
permutation of the leaf nodes to try to reduce the storage complexity of
a **TreeBasedTensor**.

Computing the Frobenius norm of a **TreeBasedTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The command **X.norm()** returns the Frobenius norm of :math:`X`.

Computing the :math:`\alpha`-singular values of a **TreeBasedTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For all :math:`\alpha \in T`, the :math:`\alpha`-singular values of
:math:`X` can be obtained with **X.singular\_values()**.

The method **rank** uses the method **singular\_values** to compute the
:math:`\alpha`-ranks, :math:`\alpha \in T`, of a **TreeBasedTensor**.

Computing the derivative of **TreeBasedTensor** with respect to one of its parameters.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For an order-\ :math:`d` tree-based tensor **X** in
:math:`{\mathbb{R}}^{N \times \cdots \times N}`,
**X.parameter\_gradient\_eval\_diag(alpha)**, for :math:`\alpha \in T`,
returns the derivative

.. math:: \left.\frac{\partial X_{i_1, \ldots, i_d}}{\partial C^\alpha}\right|_{i_1=\cdots=i_d=i}, \; i = 1, \ldots, N.

The method **parameter\_gradient\_eval\_diag** is used in the
statistical learning algorithms presented in Section
[sec:TensorLearning].

Performing operations with **TreeBasedTensor**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some operations between tensors are implemented for **TreeBasedTensor**
(see for a detailed description of the operations): the Kronecker
product with **kron**, the contraction with matrices or vectors with
**tensor\_matrix\_product** or **tensor\_vector\_product** respectively,
the evaluation of the diagonal of a contraction with matrices with
**tensor\_matrix\_product\_eval\_diag**, the dot product with **dot**.

**Z = X.tensor\_matrix\_product(M)** **tensor\_vector\_product**
**tensor\_matrix\_product\_eval\_diag** **X.kron(Y)** **X.dot(Y)**

Tensor truncation with **Truncator**
------------------------------------

The object **Truncator** embeds several methods of truncation of tensors
in different formats. Given a tolerance **tol** and a maximum rank or
tuple of ranks **r**, a **Truncator** object can be created with **t =
tensap.Truncator(tol, r)**. The thresholding type (**’hard’** or
**’soft’**) can also be specified as a third argument.

For examples of use, see the tutorial file
``tutorials\tensor_algebra\tutorial_tensor_truncation.py``.

Truncation.
^^^^^^^^^^^

The generic method **truncate** calls one of the methods presented
below, based on the type and order of its input, to obtain a truncation
of the provided tensor satisfying the relative prevision and maximal
rank requirements.

For an order :math:`2` tensor, the method **svd** is called. For a
tensor of order greater than :math:`2`, the method **hosvd** is called
for a **FullTensor**, and **hsvd** for a **TreeBasedTensor**.

Truncated singular value decomposition.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The method **svd** computes the truncated singular value decomposition
of an order :math:`2` tensor. The input tensor can be a
**numpy.ndarray**, a **tensorflow.Tensor**, a **FullTensor** or a
**CanonicalTensor**, in which case the method **trunc\_svd** is called,
or a **TreeBasedTensor**, in which case the method **hsvd** is called.

The method **trunc\_svd** computes the truncated singular value
decomposition of a matrix, with a given relative precision in Schatten
:math:`p`-norm (with a specified value for :math:`p`) and given maximal
rank. The returned truncation is a **CanonicalTensor**.

Truncated higher-order singular value decomposition.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A truncated higher-order singular value decomposition of a
**numpy.ndarray**, a **FullTensor** or a **TreeBasedTensor** can be
computed with the method **hosvd**. The output is either a
**CanonicalTensor** for an order :math:`2` tensor, or a
**TreeBasedTensor** with a trivial tree for a tensor of order greater
than :math:`2`.

Truncation in tree-based tensor format.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The method **hsvd** computes, given a **TreeBasedTensor** or a
**FullTensor** with a tree and a set of active nodes, a truncation in
tree-based tensor format.

Truncation in tensor train format.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The method **ttsvd**, given a **FullTensor**, calls the method **hsvd**
with a linear tree and all the leaf nodes inactive except the first one,
resulting in a truncation in tensor-train format.

Measures, bases and functions
=============================

**RandomVariable**
------------------

A random variable :math:`X` can be created by calling its name: for
instance, **X = tensap.UniformRandomVariable(a, b)** creates a random
variable with a uniform distribution on the interval :math:`[a, b]`. The
random variables currently implemented in tensap are:

-  **tensap.DiscreteRandomVariable(v, p)**: a random variable with
   discrete values :math:`v` and associated probabilities :math:`p`,

-  **tensap.UniformRandomVariable(a, b)**: a uniform random variable on
   :math:`[a, b]`,

-  **tensap.NormalRandomVariable(m, s)**: a normal random variable with
   mean :math:`m` and standard deviation :math:`s`,

-  **tensap.EmpiricalRandomVariable(S)**: a random variable created from
   a sample :math:`S` using kernel density estimation with Scott’s rule
   of thumb to determine the bandwidth.

A new random variable can easily be implemented in tensap by making its
class inheriting from **RandomVariable** and implementing the few
methods necessary for its creation.

Once a random variable :math:`X` is created, one can for instance
generate :math:`n` random numbers according to its distribution with
**X.random(n)**, create the orthonormal polynomials associated with its
measure with **X.orthonormal\_polynomials()** (as presented in Section
[sec:polynomials]), or evaluate its probability density function
(**X.pdf(x)**), cumulative distribution function (**X.cdf(x)**) or
inverse cumulative distribution function (**X.icdf(x)**).

**RandomVector**
----------------

A random vector :math:`X` if defined in tensap by a list of
**RandomVariable** objects and a **Copula**, characterizing the
dependencies between the random variables. Currently, only the
independent copula **IndependentCopula** is implemented.

Given a list of **RandomVariable** **random\_variables** and a **Copula
C**, a random vector can be created with **X =
tensap.RandomVector(random\_variables, copula=C)**.

Once a random vector :math:`X` is created, one can for instance generate
:math:`n` random numbers according to its distribution with
**X.random(n)**, create the orthonormal polynomials associated with its
measure with **X.orthonormal\_polynomials()** (as presented in Section
[sec:polynomials]), or evaluate its probability density function
(**X.pdf(x)**) or cumulative distribution function (**X.cdf(x)**).

**Polynomials**
---------------

Families of univariate polynomials :math:`(p_i)_{i\ge 0}` are
represented in tensap with classes inheriting from
**UnivariatePolynomials**. The :math:`i`-th polynomial :math:`p_i`
represented by a **UnivariatePolynomials** object **P** can be evaluated
with **P.polyval(x, i)**, as well as its first order derivative
(**P.d\_polyval(x, i)**) and its :math:`n`-th order derivative
(**P.dn\_polyval(x, n, i)**).

Given a measure :math:`\mu`, the moments
:math:`\int  p_{i_1}(x)...p_{i_k}(x) d\mu(x)` for
:math:`(i_1,...,i_k) \in {\mathbb{N}}^{k}` can be obtained with
**P.moment(I, mu)**, with :math:`{\texttt{\detokenize{I}}}` a
:math:`n`-by-:math:`k` array representing :math:`n` tuples
:math:`(i_1,...,i_k)`. **P.moment(I, X)** with
:math:`{\texttt{\detokenize{X}}}` a random variable considers for
:math:`\mu` the probability distribution of :math:`X`.

**CanonicalPolynomials**.
^^^^^^^^^^^^^^^^^^^^^^^^^

The family of canonical polynomials is implemented in the class
**CanonicalPolynomials**. It is such that its :math:`i`-th polynomial is
:math:`p_i(x) = x^i`.

**OrthonormalPolynomials**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Orthonormal polynomials are families of polynomials
:math:`(p_i)_{i \geq 0}` that satisfy

.. math:: \int p_i(x) p_j(x) d\mu(x)= \delta_{ij}

with :math:`\delta_{ij}` the Kronecker delta, and with :math:`\mu` some
measure.

| In tensap, the orthonormal polynomials :math:`p_i`, :math:`i \geq 0`,
  are defined using the three-term recurrence relation

  .. math::

     \begin{aligned}
                     &\tilde p_{-1}(x) = 0, \quad \tilde p_0(x) = 1, \\
                     &\tilde p_{i+1}(x) = (x - a_i)\tilde p_{i}(x) - b_i \tilde p_{i-1}(x), \quad i \geq 0,\\
                     &p_i(x) = \frac{\tilde p_i(x)}{n_i}, \quad i \geq 0
                 \end{aligned}

  with :math:`a_i` and :math:`b_i` the recurrence coefficients, and
  :math:`n_i` the norm of :math:`\tilde p_i`, defined by

  .. math::

     a_i = \frac{\int \tilde p_i(x) x \tilde p_i(x) d\mu(x)}{\int p_i(x) \tilde p_i(x) d\mu(x)}, \quad 
                     b_i = \frac{\tilde p_i(x) \tilde p_i(x) d\mu(x)}{\int \tilde p_{i-1}(x) \tilde p_{i-1}(x)  d\mu(x)}, \quad
                     n_i = \sqrt{\int\tilde p_i(x) \tilde p_i(x) d\mu(x)}.

  Implementing a new family of orthonormal polynomials in tensap is
  easy: one only needs to create a class with a method providing the
  recurrence coefficients :math:`a_i`, :math:`b_i` and the norms
  :math:`n_i`, :math:`\forall i \geq 0`.

| Are currently implemented in tensap:

-  **DiscretePolynomials**: discrete polynomials orthonormal with
   respect to the measure of a **DiscreteRandomVariable**;

-  **LegendrePolynomials**: polynomials defined on :math:`[-1,1]` and
   orthonormal with respect to the uniform measure on :math:`[-1,1]`
   with density :math:`\frac{1}{2}\mathbf{1}_{[-1,1]}(x)`;

-  **HermitePolynomials**: polynomials defined on :math:`{\mathbb{R}}`
   and orthonormal with respect to the standard gaussian measure with
   density :math:`\exp(-x^2/2)/\sqrt{2\pi}`;

-  **EmpiricalPolynomials**: polynomials orthonormal with respect to the
   measure of an **EmpiricalRandomVariable**.

| If **mu** is a **LebesgueMeasure** on :math:`[-1,1]`,
  **mu.orthonormal\_polynomials()** returns a **LegendrePolynomials**
  with suitably normalized coefficients. If **mu** is a
  **LebesgueMeasure** on :math:`[a,b]` different from :math:`[-1,1]`,
  **mu.orthonormal\_polynomials()** returns a
  **ShiftedOrthonormalPolynomials**.
| If :math:`{\texttt{\detokenize{X}}}` is a **DiscreteRandomVariable**,
  a **UniformRandomVariable**, a **NormalRandomVariable**, or a
  **EmpiricalRandomVariable**, the corresponding family of orthonormal
  polynomials can be created with the command
  **X.orthonormal\_polynomials()**. If :math:`{\texttt{\detokenize{X}}}`
  does not correspond to a default measure but can be obtained as the
  push-forward measure of a default measure by an affine transformation
  (e.g. a uniform measure on :math:`[a,b] \neq [-1,1]`, or a gaussian
  measure with mean :math:`a` and standard deviation :math:`\sigma` with
  :math:`(a,\sigma)\neq (0,1)`.), the returned object is a
  **ShiftedOrthonormalPolynomials**.

**FunctionalBasis**
-------------------

| Bases of functions can be implemented in tensap by inheriting from
  **FunctionalBasis**. The basis functions of a **FunctionalBasis**
  object **H** can be evaluated with **H.eval(x)**, as well as their
  :math:`i`-th order derivative with **H.eval\_derivative(i, x)**.
| We present below some specific bases implemented in tensap. New bases
  can easily be implemented by making their class inherit from
  **FunctionalBasis**.

**PolynomialFunctionalBasis**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The command **tensap.PolynomialFunctionalBasis** **(basis, indices)**,
with **basis** a **UnivariatePolynomials** and **indices** a list,
returns the basis of polynomials :math:`(p_i)_{i \in I}` with :math:`I`
given by **indices**.

**UserDefinedFunctionalBasis**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given a list of functions **fun**, taking each as inputs :math:`d`
variables, and a **Measure mu**, the command
**tensap.UserDefinedFunctionalBases(fun, mu, d)** returns a basis whose
functions are the ones given in **fun**, with a domain equipped with the
measure :math:`mu`.

**FullTensorProductFunctionalBasis**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **FullTensorProductFunctionalBasis** object represents a basis of
multivariate functions
:math:`\{\phi^1_{i_1}(x_1) \cdots \phi^d_{i_d}(x_d)\}_{i_1 \in I^1, \ldots, i_d \in I^d}`.
It is obtained with the command
**tensap.FullTensorProductFunctionalBasis(bases)**, where **bases** is a
list of **FunctionalBasis** or a **FunctionalBases**, containing the
different bases :math:`\{\phi^\nu_{i_\nu}\}_{i_\nu \in I^\nu}`,
:math:`\nu = 1, \ldots, d`.

**SparseTensorProductFunctionalBasis**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **SparseTensorProductFunctionalBasis** object represents a basis of
multivariate functions
:math:`\{\phi^1_{i_1}(x_1) \cdots \phi^d_{i_d}(x_d)\}_{(i_1, \ldots, i_d) \in \Lambda}`,
with :math:`\Lambda \subset I^1 \times \cdots \times I^d` a set of
multi-indices. It is obtained with the command
**tensap.SparseTensorProductFunctionalBasis(bases, indices)**, where
**bases** is a list of **FunctionalBasis** or a **FunctionalBases**,
containing the different bases
:math:`\{\phi^\nu_{i_\nu}\}_{i_\nu \in I^\nu}`,
:math:`\nu = 1, \ldots, d`, and **indices** is a **MultiIndices**
representing the set of multi-indices :math:`\Lambda`.

**FunctionalBases**
-------------------

The command **tensap.FunctionalBases(bases)**, with **bases** a list of
**FunctionalBasis**, returns an object representing a collections of
bases. To obtain a collection of :math:`d` identical bases, one can use
**tensap.FunctionalBases.duplicate(basis, d)**.

Similarly to **FunctionalBasis**, the basis functions of a
**FunctionalBases** object **H** can be evaluated with **H.eval(x)**, as
well as their :math:`i`-th order derivative with **H.eval\_derivative(i,
x)**.

**FunctionalBasisArray**
------------------------

Given a basis of functions :math:`\{\phi_{i}\}_{i \in I}`, a
**FunctionalBasisArray** object represents a function :math:`f` that
writes

.. math:: f(x) = \sum_{i \in I} a_i \phi_i(x),

with some coefficients :math:`a_i`, :math:`i \in I`, and can be created
with the command **f = tensap.FunctionalBasisArray(a, basis, shape)**,
with **shape** the output shape of :math:`f`.

A **FunctionalBasisArray** is a **Function**. It can be evaluated with
the command **f.eval(x)**, and one can obtain its derivatives with
**f.eval\_derivative(n, x)**.

**FunctionalTensor**
--------------------

Given :math:`d` bases of functions
:math:`\{\phi^\nu_{i_\nu}\}_{i_\nu \in I^\nu}`,
:math:`\nu = 1, \ldots, d`, and a tensor
:math:`a \in {\mathbb{R}}^{I^1 \times \cdots \times I^d}`, a
**FunctionalTensor** object represents a function :math:`f` that writes

.. math:: f(x) = \sum_{i_1 \in I^1} \cdots \sum_{i_d \in I^d} a_{i_1, \ldots, i_d} \phi^1_{i_1}(x_1) \cdots \phi^d_{i_d}(x_d).

The tensor :math:`a` can be in different tensor formats
(**FullTensor**, **TreeBasedTensor**, ...).

A **FunctionalTensor** is a **Function**. It can be evaluated with the
command **f.eval(x)**, and one can obtain its derivatives with
**f.eval\_derivative(n, x)**.

**Tensorizer** and **TensorizedFunction**
-----------------------------------------

| For an introduction to tensorization of functions, see
  [Ali2020ApproximationWTpartI]_, [Ali2020ApproximationWTpartII]_.
| We consider functions defined on the interval :math:`I = [0,1)`. For a
  given :math:`b \in \{2,3,\ldots, \}` and :math:`d\in {\mathbb{N}}`, an
  element :math:`x \in I` can be identified with the tuple
  :math:`(i_1,\ldots,i_d,y)`, such that

  .. math::

     \label{eq:tensorization}
                 x = t_{b,d}(i_1,\ldots,i_d,y) = \sum_{k=1}^d i_kb^{-k} + b^{-d}y

  with :math:`i_k \in I_b = \{0,\ldots,b-1\}`, :math:`k = 1,\ldots,d`,
  and :math:`y = b^d x - \lfloor b^d x \rfloor \in [0,1)`. The tuple
  :math:`(i_1,\ldots,i_d)` is the representation in base :math:`b` of
  :math:`\lfloor b^d x \rfloor`. This defines a bijective map
  :math:`t_{b,d}` from :math:`\{0,\ldots,b-1\}^{d} \times [0,1)` to
  :math:`[0,1)`.

Such a mapping is represented in tensap by the object **Tensorizer**:
**t = tensap.Tensorizer(b, d)**. For a given :math:`x` in :math:`[0,1)`,
on obtains the corresponding tuple :math:`(i_1, ..., i_d,y)` with the
command **= t.map(x)**. For a given tuple :math:`(i_1, ..., i_d,y)`, on
obtains the corresponding :math:`x` with **t.inverse\_map([i\_1, ...,
i\_d,y])**.

This identification is generalized to functions of :math:`D` variables
with **t = tensap.Tensorizer(b, d, D)**.

| The map :math:`t_{b,d}` allows to define a tensorization map
  :math:`T_{b,d}`, which associates to a univariate function :math:`F`
  defined on :math:`[0,1)` the multivariate function
  :math:`f = F \circ t_{b,d}` defined on :math:`I_b^d \times I`, such
  that

  .. math:: f(i_1,\ldots,i_d,y) = F(t_{b,d}(i_1,\ldots,i_d,y)).

  Such a function is represented in tensap by a **TensorizedFunction**,
  and can be created with **f = tensap.TensorizedFunction(fun, t)**,
  with **fun** a **function** or **Function** and **t** a
  **Tensorizer**. The **TensorizedFunction** **f** is a function of
  :math:`d+1` variables that can be evaluated with **f.eval(x)**, with
  **x** a list or **numpy.ndarray** with :math:`d+1` columns.

See the tutorial file ``tutorials\functions\tutorial_TensorizedFunction.py``.

Tools
=====

**MultiIndices**
----------------

A multi-index is a tuple :math:`(i_1,\ldots,i_d) \in {\mathbb{N}}_0^d`.
A set :math:`I \subset {\mathbb{N}}_0^d` of multi-indices is represented
with an object **MultiIndices**.

To create a multi-index set :math:`I`, we use the command
**tensap.MultiIndices(I)** with **I** a numpy array of size
:math:`\#I \times d`.

A product set :math:`I = I_1 \times \ldots \times I_d` can be obtained
with **tensap.MultiIndices.product\_set([I1,...,Id])**.

The set of multi-indices

.. math:: I = \{i  \in {\mathbb{N}}_0^d : \Vert i \Vert_{\ell^p} \le m\}

can be obtained with **tensap.MultiIndices.with\_bounded\_norm(d, p,
m)**

The set of multi-indices

.. math:: I = \{i  \in {\mathbb{N}}_0^d : i_\nu \le m_\nu , 1\le \nu \le d\}

can be obtained with **tensap.MultiIndices.bounded\_by(d, p, m)**. If
:math:`m` is of length :math:`1`, it uses :math:`m_\nu = m` for all
:math:`\nu`.

For obtaining the margin or reduced margin of an multi-index set
:math:`I`, we can use For other operations of **MultiIndices**, see the
tutorial file ``tutorials\tools\tutorial_MultiIndices.py``.

**TensorGrid**, **FullTensorGrid** and **SparseTensorGrid**
-----------------------------------------------------------

| Tensor product grids or sparse grids are represented with classes
  **FullTensorGrid** and **SparseTensorGrid**, that inherit from
  **TensorGrid**.
| See the tutorial file
  ``tutorials\functions\tutorial_functions_bases_grids.py``.

Learning
========

We present in this section some objects implemented in tensap for
learning functions or tensors.

**(Functional)TensorPrincipalComponentAnalysis**
------------------------------------------------

The objects **TensorPrincipalComponentAnalysis** (resp.
**FunctionalTensorPrincipalComponentAnalysis**) implements approximation
methods for algebraic (resp. functional) tensors based on principal
component analysis, using an adaptive sampling of the entries of the
tensor (or the function). See [nouy:2017hopca]_ for a
description of the algorithms, and for examples of use, see the tutorial
files ``tutorials\approximation\tutorial_TensorPrincipalComponentAnalysis.py``
and ``tutorials\approximation\tutorial_FunctionalTensorPrincipalComponentAnalysis.py``.

| The difference between the two objects if that
  **TensorPrincipalComponentAnalysis**\ ’ methods take as first input a
  function returning components of the algebraic tensor to learn,
  whereas the methods of **FunctionalTensorPrincipalComponentAnalysis**
  take as first input the functional tensor to learn.

Both objects are parameterized by the attributes:

-  **pca\_sampling\_factor**: a factor to determine the number of
   samples :math:`N` for the estimation of the principal components (1
   by default): if the precision is prescribed,
   :math:`N = {\texttt{\detokenize{pca_sampling_factor}}} \times N_\alpha`,
   if the rank is prescribed,
   :math:`N = {\texttt{\detokenize{pca_sampling_factor}}} \times t`;

-  **pca\_adaptive\_sampling**: a boolean indicating if adaptive
   sampling is used to determine the principal components with
   prescribed precision;

-  **tol**: an array containing the prescribed relative precision; set
   **tol = inf** for prescribing the rank;

-  **max\_rank**: an array containing the maximum alpha-ranks (the
   length depends on the format). If **len(max\_rank) == 1**, uses the
   same value for all alpha; setting **max\_rank = inf** prescribes the
   precision.

Furthermore, a **FunctionalTensorPrincipalComponentAnalysis** is
parameterized by the attributes:

-  **bases**: the functional bases used for the projection of the
   function;

-  **grid**: the **FullTensorGrid** used for the projection of the
   function on the functional bases;

-  **projection\_type**: the type of projection, the default being
   ’interpolation’.

Both objects implement four main methods:

-  **hopca**: returns the set of :math:`\{\nu\}`-principal components of
   an order :math:`d` tensor, for all :math:`\nu \in \{1,\ldots,d\}`;

-  **tucker\_approximation**: returns an approximation of a tensor of
   order :math:`d` or a function of :math:`d` variables in Tucker
   format;

-  **tree\_based\_approximation**: provided with a tree and a list of
   active nodes, returns an approximation of a tensor of order :math:`d`
   or a function of :math:`d` variables in tree-based tensor format;

-  **tt\_approximation**: returns an approximation of a tensor of order
   :math:`d` or a function of :math:`d` variables in tensor-train
   format.

**LossFunction**
----------------

| In tensap, a loss function is an object inheriting from
  **LossFunction**. Given a function **fun** and a sample as a list used
  to evaluate the loss function, a **LossFunction** object :math:`\ell`
  can be evaluated with **l.eval(fun, sample)**. The risk associated
  with **fun** can be evaluated using the sample with
  **l.risk\_estimation(fun, sample)**. Finally, the test error and
  relative test error (if defined) can be evaluated with
  **l.test\_error(fun, sample)** and **l.relative\_test\_error(fun,
  sample)**, respectively.
| Currently, three loss functions are implemented in tensap:

-  **SquareLossFunction**: :math:`\ell(g, (x, y)) = (y - g(x))^2`, used
   for least-squares regression in supervised learning, to construct an
   approximation of a random variable :math:`Y` as a function of a
   random vector :math:`X` (a predictive model);

-  **DensityL2LossFunction**: :math:`\ell(g, x) = \|g\|^2 - 2g(x)`, used
   for least-squares density estimation, to approximate the distribution
   of a random variable :math:`X` from samples of :math:`X`;

-  **CustomLossFunction**: defined by the user as any function defining
   a loss. If the loss is defined using tensorflow operations, then the
   empirical risk can be minimized using tensorflow’s automatic
   differentiation capability with a **LinearModelLearningCustomLoss**
   object, presented in the next section.

**LinearModelLearning**
-----------------------

Objects inheriting from **LinearModelLearning** implement the empirical
risk minimization associated with a linear model that writes

.. math:: g(x) = \sum_{i \in I} a_i \phi_i(x),

with :math:`\{\phi_i\}_{i \in I}` a given basis (or a set of features)
and :math:`(a_i)_{i \in I}` some coefficients, and a loss function,
introduced in the previous section.

In order to perform empirical risk minimization, a
**LinearModelLearning** object **s** must be provided with a training
sample in **s.training\_sample**. In supervised learning, for the
approximation of a random variable :math:`Y` as a function of :math:`X`,
the training sample is a list **[x, y]**, with
:math:`{\texttt{\detokenize{y}}}` represents :math:`n` samples
:math:`\{y_k\}_{k=1}^n` of :math:`Y` and :math:`x` the :math:`n`
corresponding samples :math:`\{x_k\}_{k=1}^n` of :math:`X`. In density
estimation, the training sample is an array **x** containing samples
:math:`\{x_k\}_{k=1}^n` from the distribution to estimate.

One must also provide a basis (in **s.basis**) or evaluations of the
basis on the training set (in **s.basis\_eval**, in which case the
:math:`x` are not mandatory in **s.training\_sample**). The latter
option allows for providing features :math:`\phi_i(x_k)` associated with
samples :math:`x^k`, without providing the feature maps :math:`\phi_i`.

| One can also provide the **LinearModelLearning** **s** with a test
  sample in **s.test\_data** to compute a test error.
| Currently in tensap, three different **LinearModelLearning** objects
  are implemented:

-  **LinearModelLearningSquareLoss**, to minimize the risk associated
   with a **SquareLossFunction**;

-  **LinearModelLearningDensityL2**, to minimize the risk associated
   with a **DensityL2LossFunction**;

-  **LinearModelLearningCustomLoss**, to minimize the risk associated
   with a **CustomLossFunction**.

**LinearModelLearningSquareLoss**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **LinearModelLearningSquareLoss** object **s** implements three ways
of solving the empirical risk minimization associated with a
**SquareLossFunction**:

-  by default, **s.solve()** solves the ordinary least-squares problem

   .. math:: \min_{(a_i)_{i \in I}} \frac 1 n \sum_{k=1}^n (y_k - \sum_{i \in I} a_i \phi_i(x_k))^2;

-  with the attribute **s.regularization = True**, **s.solve()** solves
   the regularized problem

   .. math:: \min_{(a_i)_{i \in I}} \frac 1 n \sum_{k=1}^n (y_k - \sum_{i \in I} a_i \phi_i(x_k))^2 + \lambda \|a\|_p

   with :math:`\lambda` a regularization hyper-parameter, selected with
   a cross-validation estimate of the error and :math:`p` specified by
   **s.regularization\_type** which can be **’l0’** (:math:`p = 0`),
   **’l1’** (:math:`p = 1`) or **’l2’** (:math:`p = 2`);

-  let us suppose that we have a collection of candidate sparsity
   patterns :math:`K_\lambda`, :math:`\lambda \in \Lambda`, for the
   parameter :math:`a`: with the attribute **s.basis\_adaptation =
   True**, **s.solve()** solves, for all :math:`\lambda\in \Lambda`, the
   problem

   .. math:: \min_{(a_i)_{i \in I}} \frac 1 n \sum_{k=1}^n (y_k - \sum_{i \in I} a_i \phi_i(x_k))^2 \quad \text{subject to } \mathrm{support}(a) \subset K_\lambda,

   where :math:`\mathrm{support}(a) = \{ k \in K : a_k \neq 0 \}`, and
   selects the optimal sparsity pattern using a cross-validation
   estimate of the error.

**LinearModelLearningDensityL2**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **LinearModelLearningDensityL2** object **s** implements three ways of
solving the empirical risk minimization associated with a
**DensityL2LossFunction**:

-  by default, **s.solve()** solves the minimization problem

   .. math:: \min_{(a_i)_{i \in I}} \| \sum_{i \in I} a_i \phi_i \|_{L^2}^2 - \frac 2 n \sum_{k=1}^n \sum_{i \in I} a_i \phi_i(x_k);

-  with the attribute **s.regularization = True**, **s.solve()** solves
   the constrained problem

   .. math:: \min_{(a_i)_{i \in I}} \|\sum_{i \in I} a_i \phi_i \|_{L^2}^2 - \frac 2 n \sum_{k=1}^n \sum_{i \in I} a_i \phi_i(x_k) \quad \text{subject to } \mathrm{support}(a) \subset K_\lambda,

   with :math:`K_\lambda`, :math:`\lambda \in \Lambda`, a sequence of
   sets of indices that introduce the coefficients solution of the
   minimization problem without regularization in decreasing order of
   magnitude. The optimal sparsity pattern is determined using a
   cross-validation estimate of the error;

-  let us suppose that we have a collection of candidate patterns
   :math:`K_\lambda`, :math:`\lambda \in \Lambda`, for the parameter
   :math:`a`: with the attribute **s.basis\_adaptation = True**,
   **s.solve()** solves, for all :math:`\lambda\in \Lambda`, the problem

   .. math:: \min_{(a_i)_{i \in I}} \|\sum_{i \in I} a_i \phi_i \|^2 - \frac 2 n \sum_{k=1}^n \sum_{i \in I} a_i \phi_i(x_k) \quad \text{subject to } \mathrm{support}(a) \subset K_\lambda,

   and selects the optimal sparsity pattern using a cross-validation
   estimate of the error.

**LinearModelLearningCustomLoss**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **LinearModelLearningCustomLoss** object **s** implements a way of
solving the empirical risk minimization associated with a
**CustomLossFunction** using tensorflow’s automatic differentiation
capabilities.

By default, the optimizer used is keras’ Adam algorithm, which is a
“stochastic gradient descent method that is based on adaptive estimation
of first-order and second-order moments” (per tensorflow’s
documentation).

The algorithm requires a starting point, provided in
**s.initial\_guess**, and several options can be set:

-  **s.options[’max\_iter’]** sets the maximum number of iterations used
   in the optimization algorithm,

-  **s.options[’stagnation’]** sets the stopping tolerance on the
   stagnation between two iterates,

-  for the Adam algorithm (and other minimization algorithms provided by
   tensorflow/keras), the learning rate can be provided in
   **s.optimizer.learning\_rate**.

**TensorLearning**
------------------

The package tensap implements algorithms to perform statistical learning
with canonical and tree-based tensor formats. See [grelier:2018]_,
[grelier2019learning]_, [2020arXiv200701165M]_ for a detailed presentation
of algorithms and related theory.

For examples, see the tutorial files
``tutorials\approximation\tutorial_tensor_learning_CanonicalTensorLearning.py``,
``tutorials\approximation\tutorial_tensor_learning_TreeBasedTensorLearning.py``,
``tutorials\approximation\tutorial_tensor_learning_TreeBasedTensorDensityLearning.py``,
``tutorials\approximation\tutorial_tensor_learning_tensorized_function_learning.py``.

These algorithms are implemented in the core object **TensorLearning**,
common to all the tensor formats, so that implementing such a learning
algorithm for a new tensor format is simple. In tensap are currently
implemented **CanonicalTensorLearning** for the learning in canonical
tensor format and **TreeBasedTensorLearning** for the learning in
tree-based tensor format.

Two algorithms are proposed: the standard one, which minimizes an
empirical risk over the set of tensors in a given format thanks to an
alternating minimization over the parameters of the tensors, and the
adaptive one, which returns a sequence of empirical risk minimizers with
adapted rank (for the canonical and tree-based tensor formats) and
adapted tree (for the tree-based tensor format).

In order to perform empirical risk minimization, a **TensorLearning**
object **s** must be provided with a training sample in
**s.training\_sample**. In supervised learning, for the approximation of
a random variable :math:`Y` as a function of :math:`X`, the training
sample is a list **[x, y]**, with :math:`{\texttt{\detokenize{y}}}`
represents :math:`n` samples :math:`\{y_k\}_{k=1}^n` of :math:`Y` and
:math:`x` the :math:`n` corresponding samples
:math:`\{x_k = (x_{k,1},\ldots,x_{k,d})\}_{k=1}^n` of :math:`X`. In
density estimation, the training sample is an array **x** containing
samples :math:`\{x_k= (x_{k,1},\ldots,x_{k,d})\}_{k=1}^n` from the
distribution to estimate.

One must also provide bases (in **s.bases**) or evaluations of the bases
on the training set (in **s.bases\_eval**, in which case the :math:`x`
are not mandatory in **s.training\_sample**). The latter option allows
for providing features :math:`\phi^\nu_i(x_{\nu,k})`,
:math:`1\le \nu\le d`, associated with samples
:math:`x_k = (x_{k,1},\ldots,x_{k,d})`, without providing the feature
maps :math:`\phi^\nu_i`.

One can also provide the **TensorLearning** **s** with a test sample in
**s.test\_data** to compute a test error.

Rank adaptation.
^^^^^^^^^^^^^^^^

(See [grelier:2018]_) The rank adaptation
is enabled by setting **s.rank\_adaptation** to **True**.

For tensors in canonical format, the algorithm returns a sequence of
rank-\ :math:`r` approximations, with
:math:`r = 1, \ldots, r_{\text{max}}`, :math:`r_{\text{max}}` being
given by **s.rank\_adaptation\_options** **[’max\_iterations’]**.

For tensors in tree-based format, the algorithm returns a sequence of
tensors with non-decreasing tree-based rank, obtained by increasing, at
each iterations, the ranks associated with a subset of nodes of the tree
:math:`T`. The number of nodes in this subset is influenced by a
parameter **s.rank\_adaptation\_options[’theta’]** in :math:`[0, 1]`,
which is such that the larger it is, the more ranks are increased at
each iteration. The default value of :math:`0.8`.

Tree adaptation.
^^^^^^^^^^^^^^^^

(See [grelier:2018]_) For tree-based
tensor formats, the tree can be adapted at each iteration using the
algorithm mentioned in Section [sec:TreeBasedTensor], by setting
**s.tree\_adaptation** to **True**. The tolerance for the tree
adaptation is provided by **s.tree\_adaptation\_options[’tolerance’]**
and the maximal number of tried trees by
**s.tree\_adaptation\_options[’max\_iterations’]**.

Model selection.
^^^^^^^^^^^^^^^^

(See [2020arXiv200701165M]_) At the end of the adaptive
procedure, a model can be selected by setting **s.model\_selection** to
**True**, using either a test error (specified by
**s.model\_selection\_options[’type’] = ’test\_error’**) or a
cross-validation estimate of the error (specified by
**s.model\_selection\_options[’type’] = ’cv\_error’**).

Example: character classification in tree-based tensor format.
--------------------------------------------------------------

We present below a part of the tutorial file
``tutorial\tensor\learning\digits\recognition.py`` shipped with the
package tensap. Its aim is to create a classifier in tree-based tensor
format, able to recognize hand written digits from :math:`0` to
:math:`9`.

The output of the algorithm is displayed below the Python script, as
well as in Figure [fig:classification\_results], which shows the
confusion matrix on the test sample as well as a visual comparison on
some test samples. We see that, using a training sample of size
:math:`1617`, it returns a classifier that obtains a score of
:math:`98.89\%` of correct classification on a test sample of size
:math:`180`.

::

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

::

    Dimension 4

    The implemented learning algorithms are designed for orthonormal bases. These algorithms work with non-orthonormal bases, but without some guarantees on their results.


    Rank adaptation, iteration 0:
        Enriched nodes: []
        Ranks = [10, 1, 1, 1, 1, 1, 1]
        Storage complexity = 144
        Test error = 9.38889e-01

    Rank adaptation, iteration 1:
        Enriched nodes: [2, 4, 3, 5, 6, 7]
        Ranks = [10, 2, 2, 2, 2, 2, 2]
        Storage complexity = 320
        Test error = 8.44444e-01

    Rank adaptation, iteration 2:
        Enriched nodes: [2, 3, 4, 5, 7]
        Ranks = [10, 3, 3, 3, 3, 2, 3]
        Storage complexity = 498
        Test error = 7.00000e-01

    Rank adaptation, iteration 3:
        Enriched nodes: [2, 3, 4, 6, 7]
        Ranks = [10, 4, 4, 4, 3, 3, 4]
        Storage complexity = 718
        Test error = 5.61111e-01

    Rank adaptation, iteration 4:
        Enriched nodes: [2, 5, 3]
        Ranks = [10, 5, 5, 4, 4, 3, 4]
        Storage complexity = 885
        Test error = 1.22222e-01

    Rank adaptation, iteration 5:
        Enriched nodes: [2, 3, 4, 5, 6, 7]
        Ranks = [10, 6, 6, 5, 5, 4, 5]
        Storage complexity = 1257
        Test error = 5.55556e-02

    Rank adaptation, iteration 6:
        Enriched nodes: [2, 3, 5, 6]
        Ranks = [10, 7, 7, 5, 6, 5, 5]
        Storage complexity = 1568
        Test error = 2.22222e-02

    Rank adaptation, iteration 7:
        Enriched nodes: [2, 3]
        Ranks = [10, 8, 8, 5, 6, 5, 5]
        Storage complexity = 1773
        Test error = 3.33333e-02

    Rank adaptation, iteration 8:
        Enriched nodes: [3, 7, 2, 6]
        Ranks = [10, 9, 9, 5, 6, 6, 6]
        Storage complexity = 2163
        Test error = 2.22222e-02

    Rank adaptation, iteration 9:
        Enriched nodes: [4, 5]
        Ranks = [10, 9, 9, 6, 7, 6, 6]
        Storage complexity = 2337
        Test error = 2.22222e-02

    Rank adaptation, iteration 10:
        Enriched nodes: [3, 4, 2, 6]
        Ranks = [10, 10, 10, 7, 7, 7, 6]
        Storage complexity = 2801
        Test error = 1.66667e-02

    Rank adaptation, iteration 11:
        Enriched nodes: [2, 3, 5]
        Ranks = [10, 11, 11, 7, 8, 7, 6]
        Storage complexity = 3212
        Test error = 2.22222e-02

    Rank adaptation, iteration 12:
        Enriched nodes: [2, 4, 6, 7, 3]
        Ranks = [10, 12, 12, 8, 8, 8, 7]
        Storage complexity = 3903
        Test error = 1.66667e-02

    Rank adaptation, iteration 13:
        Enriched nodes: [2, 3, 4, 6, 7]
        Ranks = [10, 13, 13, 9, 8, 9, 8]
        Storage complexity = 4684
        Test error = 1.66667e-02

    Rank adaptation, iteration 14:
        Enriched nodes: [5]
        Ranks = [10, 13, 13, 9, 9, 9, 8]
        Storage complexity = 4834
        Test error = 1.11111e-02

    Model selection using the test error: model #14 selected
    Ranks = [10, 13, 13, 9, 9, 9, 8], test error = 1.11111e-02
    615.6790609359741

    Accuracy = 9.88889e-01

    Classification report:
                  precision    recall  f1-score   support

               0       1.00      1.00      1.00        23
               1       1.00      0.96      0.98        23
               2       1.00      1.00      1.00        19
               3       1.00      1.00      1.00        18
               4       1.00      1.00      1.00        22
               5       0.93      1.00      0.96        13
               6       1.00      0.94      0.97        17
               7       1.00      1.00      1.00        15
               8       0.95      1.00      0.97        18
               9       1.00      1.00      1.00        12

        accuracy                           0.99       180
       macro avg       0.99      0.99      0.99       180
    weighted avg       0.99      0.99      0.99       180


    Confusion matrix:
    [[23  0  0  0  0  0  0  0  0  0]
     [ 0 22  0  0  0  0  0  0  1  0]
     [ 0  0 19  0  0  0  0  0  0  0]
     [ 0  0  0 18  0  0  0  0  0  0]
     [ 0  0  0  0 22  0  0  0  0  0]
     [ 0  0  0  0  0 13  0  0  0  0]
     [ 0  0  0  0  0  1 16  0  0  0]
     [ 0  0  0  0  0  0  0 15  0  0]
     [ 0  0  0  0  0  0  0  0 18  0]
     [ 0  0  0  0  0  0  0  0  0 12]]

.. .40 |Obtained results for the classification tutorial.|

.. .59 |Obtained results for the classification tutorial.|

.. .. |Obtained results for the classification tutorial 1.| image:: confusion_matrix
.. .. |Obtained results for the classification tutorial 2.| image:: test_sample_comparison

