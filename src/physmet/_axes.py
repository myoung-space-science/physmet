import collections.abc
import numbers
import typing

import nonstring
import numpy
import numpy.typing

from . import axis
from . import data
from . import measurable
from . import numeric


class Axes(collections.abc.Mapping, typing.Mapping[str, numeric.Axis]):
    """A mapping from dimension to axis."""
    def __init__(self, mapping: typing.Dict[str, numeric.Axis]) -> None:
        self._mapping = mapping
        self._dimensions = None
        self._names = None

    def __len__(self) -> int:
        """Called for len(self)."""
        return len(self._mapping)

    def __iter__(self) -> typing.Iterator[str]:
        """Called for iter(self)."""
        return iter(self._mapping)

    def __getitem__(self, arg: typing.Union[str, typing.SupportsIndex], /):
        """Called for self[arg]."""
        if isinstance(arg, str):
            # User requested an axis by name.
            return self._mapping[arg]
        try:
            # User requested an axis by index. Note that this relies on
            # order-preserving `dict`. Even though this became a standard
            # language feature in Python 3.7, users should prefer name-based
            # access.
            key = self.names[arg]
        except IndexError as err:
            raise KeyError(f"Unrecognized axis key: {arg}") from err
        return self._mapping[key]

    def __str__(self) -> str:
        """A printable representation of this object."""
        items = ', '.join(f"{k}: {a!r}" for k, a in self._mapping.items())
        return f"{{{items}}}"

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        items = ', '.join(f"{k}: {a!r}" for k, a in self._mapping.items())
        return f"{self.__class__.__qualname__}({items})"

    def __eq__(self, other):
        """Called for self == other."""
        if not isinstance(other, Axes):
            return NotImplemented
        if self.names != other.names:
            return False
        for k, this in self.items():
            that = other[k]
            if not numpy.array_equal(this, that):
                return False
        return True

    def __lt__(self, other):
        """Called for self < other."""
        if not isinstance(other, Axes):
            return NotImplemented
        akeys = self.keys()
        bkeys = other.keys()
        ab = akeys & bkeys
        for k in ab:
            if self[k] != other[k]:
                raise ValueError(
                    f"Cannot apply '<' to instances of {type(self)}"
                    f" with different values of {k!r}"
                ) from None
        return akeys < bkeys

    def __le__(self, other):
        """Called for self <= other."""
        if not isinstance(other, Axes):
            return NotImplemented
        akeys = self.keys()
        bkeys = other.keys()
        ab = akeys & bkeys
        for k in ab:
            if self[k] != other[k]:
                raise ValueError(
                    f"Cannot apply '<=' to instances of {type(self)}"
                    f" with different values of {k!r}"
                ) from None
        return akeys <= bkeys

    def __gt__(self, other):
        """Called for self > other."""
        if not isinstance(other, Axes):
            return NotImplemented
        akeys = self.keys()
        bkeys = other.keys()
        ab = akeys & bkeys
        for k in ab:
            if self[k] != other[k]:
                raise ValueError(
                    f"Cannot apply '>' to instances of {type(self)}"
                    f" with different values of {k!r}"
                ) from None
        return akeys > bkeys

    def __ge__(self, other):
        """Called for self >= other."""
        if not isinstance(other, Axes):
            return NotImplemented
        akeys = self.keys()
        bkeys = other.keys()
        ab = akeys & bkeys
        for k in ab:
            if self[k] != other[k]:
                raise ValueError(
                    f"Cannot apply '>=' to instances of {type(self)}"
                    f" with different values of {k!r}"
                ) from None
        return akeys >= bkeys

    def __add__(self, other):
        """Called for self + other.

        This method will fill singular dimensions in each instance with the
        corresponding full axis in the other instance, if possible.
        """
        if not isinstance(other, Axes):
            return NotImplemented
        if self.keys() != other.keys():
            raise ValueError(
                f"Cannot fill {a} from {b} with different dimensions"
            ) from None
        new = {}
        for key in self:
            a = self[key]
            b = other[key]
            if a == b:
                new[key] = a
            elif len(b) == 1 and b in a:
                new[key] = a
            elif len(a) == 1 and a in b:
                new[key] = b
            else:
                raise ValueError(
                    f"The axes {a} and {b} have inconsistent values"
                ) from None
        return axes_factory(new)

    def __or__(self, other):
        """Called for self | other."""
        if isinstance(other, typing.Mapping):
            return self._merge(other)
        return NotImplemented

    def _merge(self, other: typing.Mapping):
        """Merge self with other while preserving order."""
        these = list(self)
        those = list(other)
        try:
            keys = nonstring.merge(these, those)
        except nonstring.MergeError as err:
            errmsg = str(err).replace('entries', 'dimensions')
            raise ValueError(errmsg) from err
        mapping = {k: other[k] if k in other else self[k] for k in keys}
        return axes_factory(mapping)

    def copy(self):
        """Create a shallow copy of this collection."""
        return axes_factory({d: self[d] for d in self.dimensions})

    @typing.overload
    def replace(self, dimension: str, axis: numeric.Axis) -> typing.Self: ...

    @typing.overload
    def replace(self, dimension: str, **pair: numeric.Axis) -> typing.Self: ...

    def replace(self, *args, **pair):
        """Replace the axis object corresponding to `dimension`.

        Parameters
        ----------
        dimension : string
            The current name of the axis to replace. The given dimension must
            exist in this collection.

        axis : `~Axis`
            The new axis object that will correspond to `dimension`. May not be
            specified along with a key-value pair.

        **pair : `~Axis`
            A key-value pair in which the value is an axis object and the key is
            its corresponding name. This name will replace `dimension` in the
            result. May not be specified along with a positional axis.

        Returns
        -------
        `~Axes`
            A new instance with all the elements in this instance not
            corresponding to `dimension`, and a new dimension-axis pair
            determined from arguments.

        Raises
        ------
        `KeyError`
            The user passed a non-existent dimension.
        `ValueError`
            The user passed a positional axis and a key-value pair.
        """
        dimension, axis = args if len(args) == 2 else (args[0], None)
        if dimension not in self:
            raise KeyError(
                f"Axes do not contain {dimension!r}"
            ) from None
        if pair and axis is not None:
            raise TypeError(
                f"Got conflicting axis updates: {strargs(axis, **pair)}"
            ) from None
        if axis is not None:
            mapping = {
                d: axis if d == dimension else self[d]
                for d in self.dimensions
            }
            return axes_factory(mapping)
        if pair:
            if len(pair) > 1:
                raise ValueError(
                    "Can only replace one axis at a time."
                ) from None
            key, value = pair.popitem()
            mapping = {}
            for d in self.dimensions:
                if d == dimension:
                    mapping[key] = value
                else:
                    mapping[d] = self[d]
            return axes_factory(mapping)
        raise ValueError("Missing axis") from None

    @typing.overload
    def insert(
        self,
        dimension: str,
        new: numeric.Axis,
        index: typing.Optional[int]=None,
    ) -> typing.Self: ...

    @typing.overload
    def insert(
        self,
        dimension: str,
        new: numeric.Axis,
        before: typing.Optional[str]=None,
    ) -> typing.Self: ...

    @typing.overload
    def insert(
        self,
        dimension: str,
        new: numeric.Axis,
        after: typing.Optional[str]=None,
    ) -> typing.Self: ...

    def insert(self, dimension, new, **kwargs):
        """Insert dimension-axis pairs."""
        if len(kwargs) > 1:
            raise TypeError(
                "Got multiple location keywords for new axes"
                f": {strargs(**kwargs)}"
            ) from None
        if len(kwargs) == 0:
            return self | {dimension: new}
        locator, location = kwargs.popitem()
        mapping = {}
        for i, d in enumerate(self.dimensions):
            current = self[d]
            if locator == 'before' and d == location:
                mapping[d] = current
                mapping[dimension] = new
            elif locator == 'after' and d == location:
                mapping[dimension] = new
                mapping[d] = current
            elif locator == 'index' and i == location:
                mapping[dimension] = new
                mapping[d] = current
            else:
                mapping[d] = current
        return axes_factory(mapping)

    def without(self, dimension: str, strict: bool=False):
        """Create a set of axes without the named element.
        
        Parameters
        ----------
        dimension : string
            The name of the axis to remove.

        strict : bool, default=False
            If true and these axes do not contain an element corresponding to
            `dimension`, raise an exception. The default behavior is to ignore a
            missing dimension.

        Returns
        -------
        `~Axes`
            A new instance with all the elements in this instance, except for
            that corresponding to `dimension`.

        Raises
        ------
        `KeyError`
            The user passed a non-existent dimension with `strict = True`.
        """
        if strict and dimension not in self:
            raise KeyError(
                f"Axes does not contain {dimension!r}"
            ) from None
        mapping = {k: v for k, v in self.items() if k != dimension}
        return axes_factory(mapping)

    def extract(self, *dimensions: str):
        """Extract a subset of these axes.
        
        Parameters
        ----------
        *dimensions
            One or more axis names from which to create the new instance.

        Returns
        -------
        `~Axes`
            A new instance with elements from this instance corresponding to the
            given dimensions.

        Raises
        ------
        `~KeyError`
            These axes do not contain a given dimension.
        `~TypeError`
            The user did not provide any dimensions.
        """
        if not dimensions:
            raise TypeError(
                "No dimensions to extract"
            ) from None
        return axes_factory({d: self[d] for d in dimensions})

    @typing.overload
    def permute(
        self,
        order: typing.Iterable[typing.SupportsIndex],
    ) -> typing.Self: ...

    @typing.overload
    def permute(
        self,
        *order: typing.SupportsIndex,
    ) -> typing.Self: ...

    def permute(self, *args, **kwargs):
        """Reorder these axes."""
        if args and 'order' in kwargs:
            raise TypeError("Too many arguments") from None
        if args:
            if len(args) == 1:
                return self._permute(args[0])
            return self._permute(args)
        if 'order' in kwargs:
            return self._permute(kwargs['order'])
        raise TypeError("No arguments") from None

    def _permute(self, args: typing.Iterable):
        """Helper for `~permute`."""
        remapped = {d: self[d] for d in self._permute_dimensions(args)}
        return axes_factory(remapped)

    def _permute_dimensions(self, args: typing.Iterable):
        """Helper for `~permute`."""
        if all(isinstance(arg, typing.SupportsIndex) for arg in args):
            return self.dimensions.permute(args)
        if all(isinstance(arg, str) for arg in args):
            self.dimensions.mismatch(args)
            return data.dimensions(args)
        raise TypeError(*args)

    @property
    def dimensions(self):
        """The equivalent `~Dimensions` instance."""
        if self._dimensions is None:
            self._dimensions = data.dimensions(self.names)
        return self._dimensions

    @property
    def names(self) -> typing.Tuple[str]:
        """The axis names."""
        if self._names is None:
            self._names = tuple(self._mapping.keys())
        return self._names


def axes_factory(*args, **kwargs) -> Axes:
    """Internal factory for `~Axes` instances."""
    return Axes(_map_axes(*args, **kwargs))


def _map_axes(*args, **kwargs):
    """Create a `dict` of axes to initialize `~Axes`."""
    if not args:
        # Strip known keyword arguments from the full collection of keyword
        # arguments, leaving only dimension-axis pairs (if any).
        shape = kwargs.pop('shape', None)
        dimensions = kwargs.pop('dimensions', None)
        axes = kwargs.pop('axes', None)
        mapping = kwargs.pop('mapping', None)
        # Merge dimension-axis mappings.
        pairs = kwargs.copy()
        pairs.update(mapping or {})
        if pairs:
            # Create axes from merged dimension-axis pairs.
            return {k: convert_to_axis(arg) for k, arg in pairs.items()}
        if axes is not None:
            if dimensions is None:
                # Create axes from default dimensions and the given axes.
                return {
                    f'x{i}': convert_to_axis(axis)
                    for i, axis in enumerate(axes)
                }
            # Make sure number of dimensions == number of axes.
            na = len(axes)
            nd = len(dimensions)
            if na != nd:
                raise ValueError(
                    f"Number of dimensions ({nd}) != number of axes ({na})"
                ) from None
            # Create axes from the given dimensions and axes.
            return {
                d: convert_to_axis(axis) for d, axis in zip(dimensions, axes)
            }
        if shape is not None:
            if dimensions is None:
                # Create axes from default dimensions and the given shape.
                return {
                    f'x{i}': axis.points(range(n))
                    for i, n in enumerate(shape)
                }
            # Make sure number of dimensions == length of shape.
            ns = len(shape)
            nd = len(dimensions)
            if ns != nd:
                raise ValueError(
                    f"Number of dimensions ({nd}) != length of shape ({ns})"
                ) from None
            # Create axes from the given shape and dimensions.
            return {
                k: axis.points(range(n)) for k, n in zip(dimensions, shape)
            }
        if dimensions is not None:
            # We're here because we can't create axes from dimensions without
            # knowledge of at least the length of each axis.
            raise TypeError(
                "Cannot create axes from dimensions without shape or axes"
            ) from None
        # We're here because there are no positional arguments, there are no
        # explicit dimension-axis pairs, and all the keyword arguments are
        # `None`, meaning we have nothing from which to create axes.
        raise ValueError("Empty axes") from None
    if len(args) == 1:
        # This ignores all other arguments on purpose.
        arg = args[0]
        if isinstance(arg, typing.Mapping):
            # Create axes from the given mapping.
            return {k: convert_to_axis(a) for k, a in arg.items()}
        raise TypeError(
            f"Expected positional argument to be a mapping, not {type(arg)}"
        ) from None
    if not kwargs and all(isinstance(arg, numbers.Integral) for arg in args):
        # Create axes from one or more axis lengths. NOTE: Writing this as a
        # dict comprehension caused "UnboundLocalError: cannot access local
        # variable 'axis' where it is not associated with a value".
        d = {}
        for i, n in enumerate(args):
            d[f'x{i}'] = axis.points(range(n))
        return d
    raise TypeError(
        f"Cannot create axes from {strargs(*args, **kwargs)}"
    ) from None


def convert_to_axis(arg):
    """Attempt to convert `arg` to a valid axis object."""
    try:
        axis = _convert_to_axis(arg)
    except (TypeError, measurable.ParsingTypeError):
        raise ValueError(
            f"Cannot convert {arg} to an axis."
        ) from None
    return axis


def _convert_to_axis(arg):
    """Return a valid axis object."""
    if isinstance(arg, numeric.Axis):
        return arg
    if all(isinstance(x, str) for x in arg):
        return axis.symbols(arg)
    if all(isinstance(x, numbers.Integral) for x in arg):
        return axis.points(arg)
    m = measurable.measure(arg)
    if m.unit == '1' and all(isinstance(x, numbers.Integral) for x in m):
        return axis.points(m)
    return axis.coordinates(m)


def strargs(*pos, **kwd) -> str:
    """Format arguments as if explicitly passed."""
    def strfmt(x):
        return repr(x) if isinstance(x, str) else str(x)
    strpos = ', '.join(strfmt(x) for x in pos)
    if not kwd:
        return strpos
    strkwd = ', '.join(f"{k}={strfmt(v)}" for k, v in kwd.items())
    if not pos:
        return strkwd
    return f"{strpos}, {strkwd}"


