import numbers
import typing

import numpy
import numpy.typing

from .. import base
from .. import metric
from .. import symbolic


def isnull(this: typing.Any) -> bool:
    """True if `this` is empty but not if it's 0.

    This function allows the calling code to programmatically test for objects
    that are logically `False` except for numbers equivalent to 0.
    """
    if isinstance(this, numbers.Number):
        return False
    size = getattr(this, 'size', None)
    if size is not None:
        return size == 0
    try:
        result = not bool(this)
    except ValueError:
        result = all((isnull(i) for i in this))
    return result


def isunitlike(this):
    """True if `this` can serve as a metric unit.
    
    Notes
    -----
    This function will return `True` if `this` is

    * an instance of `~metric.Unit` or `str`, or one of their subclasses
    * a strict instance of `~symbolic.Expression`

    The justification for the restriction in the latter case is to avoid a false
    positive when `this` is a non-`Unit` subclass of `~symbolic.Expression`
    (e.g., `~metric.Dimension`).
    """
    return (
        isinstance(this, (metric.Unit, str))
        or type(this) is symbolic.Expression
    )


def hastype(
    __obj,
    __types: typing.Union[type, typing.Tuple[type, ...]],
    *wrappers: typing.Type[typing.Iterable],
    strict: bool=False,
) -> bool:
    """True if an object is a certain type or contains certain types.
    
    Parameters
    ----------
    __obj : Any
        The object to compare.

    __types : type or tuple of types
        One or more types of which the target object may be an instance.

    *wrappers : iterable type
        Zero or more iterable types of which the target object may be an
        instance. If the target object is an instance of a given wrapper type,
        this function will test whether every member of the target object is an
        instance of the given types.

    strict : bool, default=False
        If true, return `True` if `__obj` contains only members with the target
        type(s). Otherwise, return `True` if `__obj` contains at least one
        member with the target type(s).

    Examples
    --------
    When called without wrappers, this function is identical to `isinstance`:

    >>> quantity.hastype(1, int)
    True

    >>> quantity.hastype('s', str)
    True

    >>> quantity.hastype([1, 2], list)
    True

    Note that in these cases, `strict` is irrelevant because this function
    checks only the type of `__obj`.

    The target object contains the given type but `list` is not a declared
    wrapper:
    
    >>> quantity.hastype([1, 2], int)
    False
    
    Same as above, but this time `list` is a known wrapper:

    >>> quantity.hastype([1, 2], int, list)
    True
    
    Similar, except only `tuple` is declared as a wrapper:

    >>> quantity.hastype([1, 2], int, tuple)
    False

    By default, only one member of a wrapped object needs to be an instance of
    one of the target types:

    >>> quantity.hastype([1, 2.0], int, list)
    True

    If `strict=True`, each member must be one of the target types:

    >>> quantity.hastype([1, 2.0], int, list, strict=True)
    False

    Multiple target types must be passed as a `tuple`, just as when calling
    `isinstance`:

    >>> quantity.hastype([1, 2.0], (int, float), list)
    True

    Otherwise, this function will interpret them as wrapper types:

    >>> quantity.hastype([1, 2.0], int, float, list, strict=True)
    False
    """
    if isinstance(__obj, __types):
        return True
    for wrapper in wrappers:
        if isinstance(__obj, wrapper):
            check = all if strict else any
            return check(isinstance(i, __types) for i in __obj)
    return False


def isintegral(x):
    """True if is an object of integral type.

    This function exists to provide a single instance check against all integral
    types relevant to this package.
    """
    return isinstance(x, (numbers.Integral, numpy.integer))


def isordered(this, /):
    """True if `this` is an ordered quantity."""
    return isinstance(this, base.Ordered)


def iscomparable(this, /):
    """True if `this` is a comparable quantity."""
    return isinstance(this, base.Comparable)


def isadditive(this, /):
    """True if `this` is an additive quantity."""
    return isinstance(this, base.Additive)


def ismultiplicative(this, /):
    """True if `this` is a multiplicative quantity."""
    return isinstance(this, base.Multiplicative)


def isalgebraic(this, /):
    """True if `this` is an algebraic quantity."""
    return isinstance(this, base.Algebraic)


def iscomplex(this, /):
    """True if `this` is a complex-valued quantity."""
    return isinstance(this, base.Complex)


def isreal(this, /):
    """True if `this` is a real-valued quantity."""
    return isinstance(this, base.Real)


def isarray(this, /):
    """True if `this` is an array-like quantity."""
    return isinstance(this, base.Array)


def isscalar(this, /):
    """True if `this` is a scalar quantity."""
    return isinstance(this, base.Scalar)

