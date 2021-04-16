from tensap import chebyshev_points
from tensap import MultiIndices
from tensap import integer2baseb, baseb2integer

import numpy.testing as npt


def test_chebyshev_points():
    n = 10
    x = chebyshev_points(n, [-1, 1])
    print(x)
    npt.assert_almost_equal(x[0,0], 0.98768834)
    for i in range(n//2):
        npt.assert_almost_equal(x[i,0], -x[n-1-i, 0])


def test_multiindices():
    mi = MultiIndices.bounded_by([3, 5, 7])
    assert mi.cardinal() == 4*6*8, "wrong cardinal"
    mi_list = mi.to_list()
    assert len(mi_list) == mi.ndim(), "wrong list size"
    assert list(mi.get_indices(1)) == [1, 0, 0], "wrong indices"
    assert mi.is_downward_closed(), "wrong dw closed"


def test_integer2baseb():
    n = 85
    b = 3
    xb = integer2baseb(n, b)[0]
    xb_sum = sum([xb[-i-1] * b**i for i in range(len(xb))])
    assert n == xb_sum, "wrong int > base conversion"


def test_baseb2integer():
    I = [1,0,0,1,1]
    b = 3
    n = baseb2integer(I, b)
    assert n == 85, "wrong base > int conversion"
