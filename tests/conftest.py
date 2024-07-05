import numpy
import pytest

import support


@pytest.fixture
def measurables():
    """Implicitly measurable sequences."""

    unity = '1'
    unitless = [
        {'test': 1.1,    'full': (1.1, unity), 'dist': ((1.1, unity),)},
        {'test': (1.1,), 'full': (1.1, unity), 'dist': ((1.1, unity),)},
        {'test': [1.1],  'full': (1.1, unity), 'dist': ((1.1, unity),)},
        {
            'test': (1.1, 2.3),
            'full': (1.1, 2.3, unity),
            'dist': ((1.1, unity), (2.3, unity)),
        },
        {
            'test': [1.1, 2.3],
            'full': (1.1, 2.3, unity),
            'dist': ((1.1, unity), (2.3, unity)),
        },
    ]
    meter = 'm'
    withunit = [
        {'test': (1.1, meter), 'full': (1.1, meter), 'dist': ((1.1, meter),)},
        {'test': [1.1, meter], 'full': (1.1, meter), 'dist': ((1.1, meter),)},
        {
            'test': (1.1, 2.3, meter),
            'full': (1.1, 2.3, meter),
            'dist': ((1.1, meter), (2.3, meter))
        },
        {
            'test': [1.1, 2.3, meter],
            'full': (1.1, 2.3, meter),
            'dist': ((1.1, meter), (2.3, meter)),
        },
        {
            'test': [(1.1, 2.3), meter],
            'full': (1.1, 2.3, meter),
            'dist': ((1.1, meter), (2.3, meter)),
        },
        {
            'test': [[1.1, 2.3], meter],
            'full': (1.1, 2.3, meter),
            'dist': ((1.1, meter), (2.3, meter)),
        },
        {
            'test': ((1.1, meter), (2.3, meter)),
            'full': (1.1, 2.3, meter),
            'dist': ((1.1, meter), (2.3, meter)),
        },
        {
            'test': [(1.1, meter), (2.3, meter)],
            'full': (1.1, 2.3, meter),
            'dist': ((1.1, meter), (2.3, meter)),
        },
        {
            'test': [(1.1, meter), (2.3, 5.8, meter)],
            'full': (1.1, 2.3, 5.8, meter),
            'dist': ((1.1, meter), (2.3, meter), (5.8, meter)),
        },
    ]
    return [
        *unitless,
        *withunit,
    ]


@pytest.fixture
def ndarrays():
    """Base `numpy` arrays for tests."""
    r = [ # (3, 2)
        [+1.0, +2.0],
        [+2.0, -3.0],
        [-4.0, +6.0],
    ]
    xy = [ # (3, 2)
        [+10.0, +20.0],
        [-20.0, -30.0],
        [+40.0, +60.0],
    ]
    yz = [ # (2, 4)
        [+4.0, -4.0, +4.0, -4.0],
        [-6.0, +6.0, -6.0, +6.0],
    ]
    zw = [ # (4, 5)
        [+1.0, +2.0, +3.0, +4.0, +5.0],
        [-1.0, -2.0, -3.0, -4.0, -5.0],
        [+5.0, +4.0, +3.0, +2.0, +1.0],
        [-5.0, -4.0, -3.0, -2.0, -1.0],
    ]
    xyz = [ # (3, 2, 4)
        [
            [+4.0, -4.0, +4.0, -4.0],
            [-6.0, +6.0, -6.0, +6.0],
        ],
        [
            [+16.0, -16.0, +4.0, -4.0],
            [-6.0, +6.0, -18.0, +18.0],
        ],
        [
            [-4.0, +4.0, -4.0, +4.0],
            [+6.0, -6.0, +6.0, -6.0],
        ],
    ]
    return support.NDArrays(
        r=numpy.array(r),
        xy=numpy.array(xy),
        yz=numpy.array(yz),
        zw=numpy.array(zw),
        xyz=numpy.array(xyz),
    )


