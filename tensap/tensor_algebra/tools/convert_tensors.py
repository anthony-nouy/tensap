"""Tensor type conversion for binary operations.

Aligns two tensors to a common type so that operations like ``+``, ``dot``,
and ``kron`` can proceed.  Based on MATLAB's ``convertTensors``.
"""

import tensap


def convert_tensors(x, y):
    """Align two tensors to a common type for binary operations.

    Parameters
    ----------
    x : tensor object
    y : tensor object

    Returns
    -------
    (x_converted, y_converted)
        Tensors with the same type.

    Raises
    ------
    NotImplementedError
        If no conversion path exists between the two types.
    """
    _TYPES = (tensap.FullTensor, tensap.DiagonalTensor, tensap.SparseTensor,
              tensap.CanonicalTensor, tensap.TreeBasedTensor, tensap.TuckerLikeTensor)

    if not isinstance(x, _TYPES) or not isinstance(y, _TYPES):
        raise NotImplementedError(
            f"Cannot convert {type(x).__name__} and {type(y).__name__} "
            "to a common tensor type."
        )

    if type(x) is type(y):
        return x, y

    # FullTensor + other → other.full()
    if isinstance(x, tensap.FullTensor):
        return x, y.full()
    if isinstance(y, tensap.FullTensor):
        return x.full(), y

    # DiagonalTensor + TreeBasedTensor → tree_based_tensor(diag, tree)
    if isinstance(x, tensap.DiagonalTensor) and isinstance(y, tensap.TreeBasedTensor):
        return x.tree_based_tensor(y.tree), y
    if isinstance(y, tensap.DiagonalTensor) and isinstance(x, tensap.TreeBasedTensor):
        return x, y.tree_based_tensor(x.tree)

    # SparseTensor + DiagonalTensor → sparse(diag)
    if isinstance(x, tensap.SparseTensor) and isinstance(y, tensap.DiagonalTensor):
        return x, y.sparse()
    if isinstance(y, tensap.SparseTensor) and isinstance(x, tensap.DiagonalTensor):
        return x.sparse(), y

    # CanonicalTensor + TuckerLikeTensor → TLT(canonical)
    if isinstance(x, tensap.CanonicalTensor) and isinstance(y, tensap.TuckerLikeTensor):
        x = tensap.TuckerLikeTensor(x.core, x.space)
        return convert_tensors(x, y)
    if isinstance(y, tensap.CanonicalTensor) and isinstance(x, tensap.TuckerLikeTensor):
        y = tensap.TuckerLikeTensor(y.core, y.space)
        return convert_tensors(x, y)

    # TuckerLikeTensor + TuckerLikeTensor — align cores
    if isinstance(x, tensap.TuckerLikeTensor) and isinstance(y, tensap.TuckerLikeTensor):
        cx, cy = convert_tensors(x.core, y.core)
        return tensap.TuckerLikeTensor(cx, x.space), tensap.TuckerLikeTensor(cy, y.space)

    raise NotImplementedError(
        f"Cannot convert {type(x).__name__} and {type(y).__name__} "
        "to a common tensor type."
    )
