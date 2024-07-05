import numbers
import typing

import numpy
import numpy.typing

from ._implementations import (
    coordinates_factory as coordinates,
    points_factory as points,
    symbols_factory as symbols,
)
from ._types import (
    Coordinates,
    Points,
    Symbols,
)

__all__ = [
    Coordinates,
    Points,
    Symbols,
    coordinates,
    points,
    symbols,
]


def restrict(
    array: numpy.ndarray,
    *targets: float,
    axis: typing.SupportsIndex,
) -> range:
    """Restrict indices of `array` around the target value(s) along `axis`.

    Parameters
    ----------
    array : array-like
        The array containing `target`.

    *targets : real
        The numerical value(s) in `array` whose index to bound. There must be at
        least one target value.

    axis : integral, default=-1
        The axis in `array` that contains the target value(s).

    Returns
    -------
    `range`

    Raises
    ------
    `ValueError`:
        The caller did not pass any target values.
    """
    if len(targets) == 0:
        raise ValueError("No target values to bound") from None
    if len(targets) == 1:
        lower, upper = _compute_bounds(array, targets[0], axis)
        return range(lower, upper+1)
    lower = None
    upper = None
    for target in targets:
        lo, hi = _compute_bounds(array, target, axis)
        lower = lo if lower is None else min(lower, lo)
        upper = hi if upper is None else max(upper, hi)
    return range(lower, upper+1)


def _compute_bounds(
    array: numpy.ndarray,
    target: float,
    axis: typing.SupportsIndex,
) -> typing.Tuple[int, int]:
    """Compute the indices in `array` that bound `target` along `axis`.

    This function is a helper for `~restrict`.
    """
    bounds = _get_index_bounds(array, target, axis=axis)
    if bounds.ndim == 1:
        size = bounds.size
        if size != 2:
            raise ValueError(
                f"Attempt to bound {target} along axis {axis} in {array}"
                f" produced a {size}-element array."
            ) from None
        return tuple(bounds)
    if bounds.ndim == 2:
        lower = numpy.min(bounds[:, 0])
        upper = numpy.max(bounds[:, 1])
        return lower, upper
    # NOTE: We can get `lower` and `upper` for N-D arrays by replacing `:` with
    # `...`. The problem is that they will be (N-1)-D arrays that we will then
    # need to split into individual index vectors before returning.
    raise NotImplementedError(
        f"Cannot compute bounds of {bounds.ndim}-D array"
    ) from None


def _get_index_bounds(
    array: numpy.typing.ArrayLike,
    target: numbers.Real,
    axis: typing.SupportsIndex=None,
) -> numpy.typing.NDArray[numpy.integer]:
    """Find the indices bounding the target value.

    This function is a helper for `~restrict`. It returns an array containing
    the indices bounding `target` at each slice along `axis`. The shape will be
    the same as that of `array` except for `axis`, which will be 2.
    """
    if axis is None:
        axis = -1
    return numpy.apply_along_axis(
        _find_1d_indices,
        int(axis),
        numpy.asfarray(array),
        float(target),
    )


def _find_1d_indices(
    array: numpy.ndarray,
    target: float,
) -> typing.Tuple[int, int]:
    """Find the bounding indices in a 1-D array."""
    leq = array <= target
    lower = numpy.where(leq)[0].max() if any(leq) else 0
    geq = array >= target
    upper = numpy.where(geq)[0].min() if any(geq) else len(array)-1
    return lower, upper


