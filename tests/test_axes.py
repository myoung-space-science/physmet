import itertools
import numbers
import typing

import numpy
import numpy.typing
import pytest

import physmet


def test_axes_factory():
    """Test ways to initialize axes."""
    shape = (3, 4)
    dimensions = ('x', 'y')
    pairs = {d: numpy.arange(i) for (d, i) in zip(dimensions, shape)}
    axes = pairs.values()
    # Initialize from a shape.
    check_init_from_shape(shape)
    check_init_from_shape(shape, dimensions=dimensions)
    # Initialize from axes.
    check_init_from_axes(axes)
    check_init_from_axes(axes, dimensions=dimensions)
    # Initialize from key-value pairs.
    check_init_from_pairs(pairs)
    # Check error cases.
    errors = (
        # cannot pass shape by position
        ((shape,), {}, TypeError),
        # cannot create axes from dimensions without shape or axes
        ((dimensions,), {}, TypeError),
        (dimensions, {}, TypeError),
        ((), {'dimensions': dimensions}, TypeError),
        # length of dimensions must equal length of shape
        ((), {'shape': shape, 'dimensions': dimensions[:-1]}, ValueError),
        # cannot pass axes by position or variable position
        ((axes,), {}, TypeError),
        (axes, {}, TypeError),
        # length of dimensions must equal length of axes
        ((), {'axes': axes, 'dimensions': dimensions[:-1]}, ValueError),
        # cannot create empty axes
        ((), {}, ValueError),
    )
    for args, kwargs, error in errors:
        with pytest.raises(error):
            physmet.axes(*args, **kwargs)


def check_init_from_shape(
    shape: typing.Sequence[numbers.Integral],
    dimensions: typing.Optional[typing.Sequence[str]]=None,
) -> None:
    """Helper for `test_axes_factory`."""
    if dimensions is not None:
        cases = (
            physmet.axes(shape=shape, dimensions=dimensions),
        )
    else:
        cases = (
            physmet.axes(*shape),
            physmet.axes(shape=shape),
        )
        dimensions = ('x0', 'x1')
    for axes in cases:
        assert len(axes) == len(shape)
        assert tuple(axes) == tuple(dimensions)
        for key, n in zip(dimensions, shape):
            vector = axes[key]
            assert len(vector) == n
            assert vector == physmet.axis.points(range(n))


def check_init_from_axes(
    axes: typing.Iterable,
    dimensions: typing.Optional[typing.Sequence[str]]=None,
) -> None:
    """Helper for `test_axes_factory`."""
    if dimensions is not None:
        these = physmet.axes(axes=axes, dimensions=dimensions)
    else:
        dimensions = ('x0', 'x1')
        these = physmet.axes(axes=axes)
    assert len(these) == len(axes)
    assert tuple(these) == tuple(dimensions)
    for dimension, axis in zip(dimensions, axes):
        vector = these[dimension]
        assert len(vector) == len(axis)
        assert vector == physmet.axis.points(axis)


def check_init_from_pairs(pairs: dict):
    """Helper for `test_axes_factory`."""
    ignored = {
        'shape': (2, 8),
        'dimensions': ('A', 'B'),
        'axes': (range(7), range(9)),
    }
    cases = (
        physmet.axes(**pairs),
        physmet.axes(mapping=pairs, **ignored),
        physmet.axes(pairs),
        physmet.axes(pairs, **ignored)
    )
    for axes in cases:
        assert len(axes) == len(pairs)
        assert tuple(axes) == tuple(pairs)
        for dimension, axis in pairs.items():
            assert axes[dimension] == physmet.axis.points(axis)


def test_axes_comparisons():
    """Test comparative operations between axes objects."""
    x = numpy.arange(4)
    y = numpy.arange(5)
    z = numpy.arange(1, 5)
    w = numpy.arange(1, 6)
    xy = physmet.axes(x=x, y=y)
    xyz = physmet.axes(x=x, y=y, z=z)
    xyzw = physmet.axes(x=x, y=y, z=z, w=w)
    assert xy < xyz
    assert xyz > xy
    assert xyz < xyzw
    assert xyzw > xyz
    assert xy <= xyz
    assert xyz >= xy
    assert xyz <= xyzw
    assert xyzw >= xyz


def test_axes_add():
    """Test the operation that fills in singular dimensions."""
    x = numpy.arange(3)
    y = numpy.arange(4)
    z = numpy.arange(5)
    full = physmet.axes(x=x, y=y, z=z)
    valid = (
        (full, physmet.axes(x=[1], y=y, z=z)),
        (full, physmet.axes(x=x, y=[1], z=z)),
        (full, physmet.axes(x=x, y=y, z=[1])),
        (full, physmet.axes(x=x, y=[1], z=[1])),
        (full, physmet.axes(x=[1], y=y, z=[1])),
        (full, physmet.axes(x=[1], y=[1], z=z)),
        (full, physmet.axes(x=[1], y=[1], z=[1])),
    )
    # for a, b in valid:
    #     assert a + b == full
    #     assert b + a == full
    errors = (
        # (full, physmet.axes(x=[10], y=y, z=z)),
        (full, physmet.axes(x=[1.0], y=y, z=z)),
        (full, physmet.axes(x=[1, 2], y=y, z=z)),
    )
    for a, b in errors:
        with pytest.raises(ValueError):
            a + b
        with pytest.raises(ValueError):
            b + a


def test_axes_merge():
    """Test merging operations between axes objects."""
    x = numpy.arange(4)
    y = numpy.arange(5)
    z = numpy.arange(1, 5)
    w = numpy.arange(1, 6)
    axbx = physmet.axes(a=x, b=x)
    axby = physmet.axes(a=x, b=y)
    axcz = physmet.axes(a=x, c=z)
    czdw = physmet.axes(c=z, d=w)
    assert axbx | axby == physmet.axes(a=x, b=y)
    assert axby | axcz == physmet.axes(a=x, b=y, c=z)
    assert axby | czdw == physmet.axes(a=x, b=y, c=z, d=w)
    assert czdw | axby != physmet.axes(a=x, b=y, c=z, d=w)


def test_axes_copy():
    """Test the ability to copy an axes collection."""
    x = numpy.arange(4)
    y = numpy.arange(5)
    old = physmet.axes(x=x, y=y)
    new = old.copy()
    assert old == new
    assert old is not new


def test_axes_replace():
    """Test the special method for replacing an axis."""
    x = numpy.arange(4)
    y = numpy.arange(5)
    z = numpy.arange(1, 5)
    old = physmet.axes(x=x, y=y)
    assert old.replace('y', z) == physmet.axes(x=x, y=z)
    assert old.replace('y', z=z) == physmet.axes(x=x, z=z)


def test_axes_insert():
    """Test the special method for inserting axes."""
    x = numpy.arange(4)
    y = numpy.arange(5)
    z = numpy.arange(1, 5)
    xz = physmet.axes(x=x, z=z)
    xy = physmet.axes(x=x, y=y)
    xyz = physmet.axes(x=x, y=y, z=z)
    assert xz.insert('y', y, index=1) == xyz
    assert xz.insert('y', y, before='x') == xyz
    assert xz.insert('y', y, after='z') == xyz
    assert xy.insert('z', z) == xyz
    with pytest.raises(TypeError):
        xz.insert(index=0, before='x', after='x', y=y)


def test_axes_without():
    """Test the special method for removing an axis."""
    x = numpy.arange(4)
    y = numpy.arange(5)
    z = numpy.arange(1, 5)
    xy = physmet.axes(x=x, y=y)
    xz = physmet.axes(x=x, z=z)
    xyz = physmet.axes(x=x, y=y, z=z)
    assert xyz.without('z') == xy
    assert xyz.without('y') == xz
    assert xyz.without('a') == xyz
    with pytest.raises(KeyError):
        xyz.without('a', strict=True)


def test_axes_extract():
    """Test the special method for extracting a subset of axes."""
    x = numpy.arange(4)
    y = numpy.arange(5)
    z = numpy.arange(1, 5)
    xy = physmet.axes(x=x, y=y)
    xz = physmet.axes(x=x, z=z)
    xyz = physmet.axes(x=x, y=y, z=z)
    assert xyz.extract('x', 'y') == xy
    assert xyz.extract('x', 'z') == xz
    with pytest.raises(KeyError):
        xyz.extract('a', 'b')
    with pytest.raises(TypeError):
        xyz.extract()


def test_axes_permute():
    """Test the ability to permute the order of axes."""
    mapping = {
        'x': numpy.arange(4),
        'y': numpy.arange(5),
        'z': numpy.arange(1, 5),
    }
    names = list(mapping)
    old = physmet.axes(mapping)
    for permutation in itertools.permutations(names):
        indices = [names.index(i) for i in permutation]
        remapped = {k: mapping[k] for k in permutation}
        expected = physmet.axes(remapped)
        for order in (indices, permutation):
            assert old.permute(order) == expected
            assert old.permute(*order) == expected
            assert old.permute(order=order) == expected


