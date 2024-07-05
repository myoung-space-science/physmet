import abc
import collections.abc
import contextlib
import numbers
import typing

import aliasedkeys

from .. import symbolic
from . import _reference


class UnitParsingError(Exception):
    """Error when attempting to parse string into unit."""

    def __init__(self, string: str) -> None:
        self.string = string

    def __str__(self) -> str:
        return f"Could not determine unit and magnitude of '{self.string}'"


class UnitConversionError(Exception):
    """Unknown unit conversion."""


class SystemAmbiguityError(Exception):
    """The metric system is ambiguous."""


class UnitSystemError(Exception):
    """The metric system does not contain this unit."""


class QuantityError(Exception):
    """An error occurred in `~metric.Quantity`."""


T = typing.TypeVar('T')


class Unit(symbolic.Expression):
    """A symbolic expression representing a physical unit."""

    def __init__(self, terms: typing.List[symbolic.Term]) -> None:
        super().__init__(terms)
        self._dimensions = None
        self._decomposed = None
        self._dimensionless = None
        self._quantity = None
        self._norms = dict.fromkeys(_reference.SYSTEMS)

    def normalize(self, system: str, quantity: str=None):
        """Represent this unit in base units of `system`.

        This is an instance-method version of `~normalize`. See that function
        for full documentation.
        """
        return normalize(self, system, quantity)

    def __mul__(self, other):
        """Called for self * other."""
        if isinstance(other, numbers.Real):
            if other == 1:
                return self
            raise ValueError(f"Cannot compute {self!r} * {other}") from None
        if self is other or super().__eq__(other):
            return unit_factory(super().__pow__(2))
        return _apply_operator(symbolic.product, self, other)

    def __rmul__(self, other):
        """Called for other * self."""
        if isinstance(other, numbers.Real):
            if other == 1:
                return self
            raise ValueError(f"Cannot compute {other} * {self!r}") from None
        return _apply_operator(symbolic.product, other, self)

    def __truediv__(self, other):
        """Called for self / other."""
        if isinstance(other, numbers.Real):
            if other == 1:
                return self
            raise ValueError(f"Cannot compute {self!r} / {other}") from None
        if self is other or super().__eq__(other):
            return unit_factory(1)
        return _apply_operator(symbolic.ratio, self, other)

    def __rtruediv__(self, other):
        """Called for other / self."""
        if isinstance(other, numbers.Real):
            if other == 1:
                return _apply_operator(symbolic.ratio, '1', self)
            raise ValueError(f"Cannot compute {other} / {self!r}") from None
        return _apply_operator(symbolic.ratio, other, self)

    def __pow__(self, other):
        """Called for self ** other."""
        if not isinstance(other, numbers.Real):
            return NotImplemented
        return unit_factory([term ** other for term in self])

    def __rpow__(self, other):
        """Called for other ** self."""
        return NotImplemented

    def __rshift__(self, other):
        """Compute the numerical factor to convert this unit to `other`.

        Examples
        --------
        Create two unit objects representing a meter and a centimeter, and
        compute their relative magnitude::

        >>> m = Unit('m')
        >>> cm = Unit('cm')
        >>> cm >> m
        0.01
        >>> m >> cm
        100.0

        As with `~metric.NamedUnit`, these results are equivalent to
        the statement that there are 100 centimeters in a meter. However, this
        class also supports more complex unit expressions, and can therefore
        compute more complex ratios::

        >>> Unit('g * m^2 / s^2') >> Unit('kg * m^2 / s^2')
        0.001
        >>> Unit('g * mm^2 / s^2') >> Unit('kg * m^2 / s^2')
        1e-09
        >>> Unit('g * m^2 / day^2') >> Unit('kg * m^2 / s^2')
        1.3395919067215364e-13
        >>> Unit('g * au^2 / day^2') >> Unit('kg * m^2 / s^2')
        2997942777.7207007
        """
        if not isinstance(other, (str, Unit)):
            return NotImplemented
        return float(conversion_factory(str(self), str(other)))

    def __rrshift__(self, other):
        """Reflected version of `__rshift__`."""
        if isinstance(other, str):
            return float(conversion_factory(other, str(self)))
        return NotImplemented

    def __lshift__(self, other):
        """Compute the numerical factor to convert `other` to this unit."""
        if not isinstance(other, (str, Unit)):
            return NotImplemented
        return float(conversion_factory(str(other), str(self)))

    def __rlshift__(self, other):
        """Reflected version of `__lshift__`."""
        if isinstance(other, str):
            return float(conversion_factory(str(self), other))
        return NotImplemented

    def __eq__(self, other) -> bool:
        """Called for self == other.

        Two unit expressions are equal if they satisfy one of the following
        conditions:

        1. are identical
        1. have symbolically equal strings
        1. have symbolically equivalent strings

        For example, all of the following are true:

        * `Unit('N') == Unit('N')` (condition 1)
        * `Unit('m / s') == 'm / s'` (condition 2)
        * `'m / s' == Unit('m / s')` (condition 2)
        * `Unit('m / s') == 'm s^-1'` (condition 3)
        * `'m / s' == Unit('m s^-1)'` (condition 3)

        See Also
        --------
        `__and__`:
            Test equivalence of two unit expressions.
        `__or__`:
            Test consistency of two unit expressions.
        """
        if other is self:
            # If they are identical, they are equal.
            return True
        # Otherwise, they are equal iff their symbolic expressions are equal.
        try:
            return super().__eq__(other)
        except AttributeError as err:
            raise AttributeError(
                f"{err}. A common cause of this is creating a unit"
                " via the class initializer (i.e., metric.Unit)"
                " rather than via the factory function (i.e., metric.unit)."
            ) from None

    def __and__(self, other):
        """Called for self & other.

        This method tests whether the given unit is equivalent to this unit. Two
        unit expressions are are equivalent if they satisfy one of the following
        conditions:

        1. are equal (see docstring at `__eq__`)
        1. differ only by dimensionless terms
        1. have a ratio of unity

        For example, all of the following are true:

        * `Unit('1 / s') & '# / s'` (condition 2)
        * `'1 / s' & Unit('# / s')` (condition 2)
        * `Unit('N') & 'kg * m / s^2'` (condition 3)
        * `'N' & Unit('kg * m / s^2')` (condition 3)

        Equivalence thus defines a notion of equality that does not require two
        unit expressions that represent equal physical quantities to have equal
        lexical representations.
        """
        if self == other:
            return True
        unity = (
            str(term) in _reference.UNITY
            for term in self.difference(other, symmetric=True)
        )
        if all(unity):
            # If the only terms that differ between the units are dimensionless
            # terms, we can declare them equal by inspection (i.e., without
            # computing their conversion factor).
            return True
        that = unit_factory(other)
        if self.dimensionless != that.dimensionless:
            # If one, and only one, of the units is dimensionless, they can't
            # possibly be equivalent.
            return False
        if set(self.decomposed) == set(that.decomposed):
            # If their base units are equal, the units are equivalent.
            return True
        if symbolic.equality(self.decomposed, that.decomposed):
            # If their base units produce equal expressions, the units are
            # equivalent.
            return True
        with contextlib.suppress(UnitConversionError):
            # If their numerical conversion factor is unity, they are
            # equivalent.
            return self << that == 1.0
        return False

    __rand__ = __and__
    """Called for other & self.

    Notes
    -----
    This is the reflected version of `~metric.Unit.__and__`. It exists to
    support equivalence comparisons between instances of `~metric.Unit` and
    objects of other types for which that comparison is meaningful, in cases
    where the other object is the left operand. The semantics are identical.
    """

    def __or__(self, other):
        """Called for self | other.

        This method tests whether the given unit is metrically consistent with
        this unit. Two units are metrically consistent if they satisfy one of
        the following conditions:

        1. are equal (see docstring at `__eq__`)
        1. are equivalent (see docstring at `__and__`)
        1. have the same dimension in at least one metric system
        1. have a defined conversion factor

        For example, all of the following are true:

        * `Unit('J') | 'erg'` (condition 3)
        * `'J' | Unit('erg')` (condition 3)
        * `Unit('J') | 'eV'` (condition 4)
        * `'J' | Unit('eV')` (condition 4)

        Consistency thus provides a way to determine if this unit can convert to
        the given unit without necessarily attempting the conversion.
        """
        if self == other or self & other:
            return True
        that = unit_factory(other)
        for system in _reference.SYSTEMS:
            defined = self.dimensions[system]
            given = that.dimensions.values()
            if defined and any(d == defined for d in given):
                return True
        if _equivalent_quantities(self, that):
            return True
        try:
            self << that
        except UnitConversionError:
            return False
        else:
            return True

    __ror__ = __or__
    """Called for other | self.

    Notes
    -----
    This is the reflected version of `~metric.Unit.__or__`. It exists to support
    consistency comparisons between instances of `~metric.Unit` and objects of
    other types for which that comparison is meaningful, in cases where the
    other object is the left operand. The semantics are identical.
    """

    @property
    def quantity(self):
        """A symbolic expression of this unit's metric quantity."""
        if self._quantity is None:
            self._quantity = get_quantity(self)
        return self._quantity

    @property
    def dimensionless(self):
        """True if this unit's dimension is '1' in all metric systems.
        
        Notes
        -----
        This property exists as a convenient shortcut for comparing this unit's
        dimension in each metric system to '1'. If you want to check whether
        this unit is dimensionless in a specific metric system, simply check the
        `dimensions` property for that system.
        """
        if self._dimensionless is None:
            self._dimensionless = all(
                dimension == '1'
                for dimension in self.dimensions.values()
            )
        return self._dimensionless

    @property
    def dimensions(self):
        """The physical dimension of this unit in each metric system."""
        if self._dimensions is None:
            mapping = {}
            for s in _reference.SYSTEMS:
                try:
                    r = self._dimensions_in(s)
                except ValueError:
                    mapping[s] = None
                else:
                    mapping[s] = r
            self._dimensions = Dimensions(**mapping)
        return self._dimensions

    def _dimensions_in(self, system: typing.Literal['mks', 'cgs']):
        """Internal helper for `dimensions` property."""
        expression = symbolic.expression('1')
        systems = set()
        for term in self:
            named = NamedUnit(term.base)
            allowed = named.systems['allowed']
            dimension = (
                named.dimensions[system] if len(allowed) > 1
                else named.dimensions[allowed[0]]
            )
            expression *= symbolic.expression(dimension) ** term.exponent
            systems.update(allowed)
        if system in systems:
            return Dimension(expression)
        raise ValueError(
            f"Can't define dimension of {self!r} in {system!r}"
        ) from None

    @property
    def decomposed(self):
        """This unit's decomposition into base units, where possible."""
        if self._decomposed is None:
            self._decomposed = _decompose_unit(self)
        return self._decomposed


def _apply_operator(f, a, b):
    """Compute `f(a, b)` where at least `a` or `b` is a `~Unit`.

    This method will attempt to reduce each operand into base units before
    computing the result, in order to reduce the result as much as possible
    """
    if unitlike(a) and unitlike(b):
        return unit_factory(f(_decompose_unit(a), _decompose_unit(b)))
    if isinstance(a, Unit) and isinstance(b, numbers.Real):
        return unit_factory(f(a.decomposed, b))
    if isinstance(a, numbers.Real) and isinstance(b, Unit):
        return unit_factory(f(a, b.decomposed))
    return NotImplemented


def _decompose_unit(unit: typing.Union[str, Unit]):
    """Decompose the given unit into base units where possible."""
    decomposed = [
        part
        for term in symbolic.expression(unit)
        for part in _decompose_term(term)
    ]
    return symbolic.reduce(decomposed)


def _decompose_term(term: symbolic.Term):
    """Decompose a symbolic term into named units, if possible."""
    try:
        # Possible outcomes
        # - success: returns list of terms
        # - decomposition failed: returns `None`
        # - parsing failed: raises `UnitParsingError`
        # - metric system is ambiguous: raises `SystemAmbiguityError`
        current = NamedUnit(term.base).decomposed
    except (UnitParsingError, SystemAmbiguityError):
        # This effectively reduces the three failure modes listed above into
        # one result.
        current = None
    if current is None:
        # If the attempt to decompose this unit term failed or raised an
        # exception, our only option is to append the existing term to the
        # running list.
        return [term]
    # If the attempt to decompose this unit term succeeded, we need to
    # distribute the term's exponent over the new terms and append each to
    # the running list.
    return [base**term.exponent for base in current]


_UNITS = aliasedkeys.MutableMapping()
"""Internal collection of singleton `~Unit` instances."""

def unit_factory(arg: typing.Union[str, Unit]) -> Unit:
    """Create a metric unit representing the given expression."""
    if isinstance(arg, Unit):
        return arg
    # Attempt to extract a string representing a single unit.
    if isinstance(arg, str):
        string = arg
    else:
        try:
            string = str(arg[0]) if len(arg) == 1 else None
        except TypeError:
            string = None
    if available := _UNITS.get(string):
        # If the argument maps to an existing unit, return that unit.
        return available
    # First time through: create a symbolic expression from `arg`.
    expression = symbolic.expression(arg)
    # The canonical string representation of this unit is the expression
    # string of its unit symbols. For example, both 'm / s' and 'meter /
    # second' should map to 'm / s'.
    symbols = [_as_symbol(term) for term in expression]
    instance = Unit(symbols)
    name = str(instance)
    if name in _UNITS:
        # It turns out that the argument corresponds to an existing unit.
        existing = _UNITS[name]
        if isinstance(arg, str):
            # If the argument is a string, register the argument as an alias
            # for that unit so we can just retrieve it next time.
            _UNITS.alias(name, arg)
        return existing
    # Create the initial mapping aliases for this unit.
    try:
        # If `name` corresponds to a named unit, register both the name and
        # symbol (e.g., 'centimeter' and 'cm').
        this = NamedUnit(name)
        key = (this.name, this.symbol)
    except UnitParsingError:
        # If attempting to parse a named unit from `name` failed, register
        # the canonical string and, if applicable, the string argument.
        key = (name, string) if string else name
    # Store and return the new instance.
    _UNITS[key] = instance
    return instance


def normalize(
    unit: typing.Union[str, Unit],
    system: str,
    quantity: str=None,
) -> Unit:
    """Represent a unit in base units of `system`.
    
    Parameters
    ----------
    unit : string or `~Unit`
        The original unit. If `unit` is a string, this function will first
        convert it to an instance of `~Unit`.

    system : string
        The metric system in which to express the result.

    quantity : string, optional
        The physical quantity associated with `unit`. If `quantity` is omitted,
        this function will compute a suitable quantity from the base units of
        `unit`. Providing an explicit quantity may resolve ambiguity in certain
        cases (e.g., between 'velocity' and 'conductance' in CGS).

    Returns
    -------
    `~Unit`
    """
    this = unit_factory(unit)
    terms = (
        get_quantity(this) if quantity is None
        else symbolic.expression(quantity)
    )
    expression = [
        symbolic.power(UNITS[term.base][system], term.exponent)
        for term in terms
    ]
    return unit_factory(expression)


def _as_symbol(unit: symbolic.Term):
    """Represent `unit` in terms of its symbol."""
    this = _reference.NAMED_UNITS[unit.base]
    prefix = this['prefix']['symbol']
    base = this['base']['symbol']
    return symbolic.term(prefix + base) ** unit.exponent


def _equivalent_quantities(u0: Unit, u1: Unit):
    """True if two units have equivalent physical quantities."""
    try:
        q0 = get_quantity(u0)
        q1 = get_quantity(u1)
    except Exception:
        # NOTE: This is the same pattern that `numpy.array_equal` uses.
        # Basically, if anything goes wrong while trying to compute the physical
        # quantity of either unit, we give up and declare them not equivalent.
        return False
    return q0 == q1


UnitType = typing.TypeVar('UnitType', Unit, symbolic.Expressable)
"""Type variable for unit-like objects.

A unit-like object is an object that can represent a metric unit, if it
corresponds to a known metric unit. Formally, it is an object that can
instantiate the `~metric.Unit` class. This definition makes no guarantee that
the resulting instance will be valid.

See Also
--------
`~metric.UnitLike`
    The equivalent type alias.
`~metric.unitlike`
    The corresponding instance test.
"""


UnitLike = typing.Union[Unit, symbolic.Expressable]
"""Type alias for unit-like objects.

A unit-like object is an object that can represent a metric unit, if it
corresponds to a known metric unit. Formally, it is an object that can
instantiate the `~metric.Unit` class. This definition makes no guarantee that
the resulting instance will be valid.

See Also
--------
`~metric.UnitType`
    The equivalent type variable.
`~metric.unitlike`
    The corresponding instance test.
"""


def unitlike(this):
    """True if `this` can serve as a metric unit.
    
    Notes
    -----
    This function will return `True` if `this` is

    * an instance of `~metric.Unit` or `str`, or one of their subclasses
    * a strict instance of `~symbolic.Expression`

    The justification for the restriction in the latter case is to avoid a false
    positive when `this` is a `~metric` subclass of `~symbolic.Expression`
    (e.g., `~metric.Dimension`).

    See Also
    --------
    `~metric.UnitType`
        The type variable for unit-like objects.
    `~metric.UnitLike`
        The type alias for unit-like objects.
    """
    return (
        isinstance(this, (Unit, str))
        or type(this) is symbolic.Expression
    )


def reduction(unit: symbolic.Expressable, system: str=None):
    """Reduce the given unit expression, if possible.
    
    Notes
    -----
    This function is still experimental.
    """
    expression = symbolic.expression(unit)
    decomposed = []
    for term in expression:
        try:
            # NOTE: `NamedUnit.reduce` can return `None`
            current = NamedUnit(term.base).reduce(system=system)
        except (UnitParsingError, SystemAmbiguityError):
            current = None
        if current:
            decomposed.append(current**term.exponent)
    if not decomposed:
        return
    result = decomposed[0]
    for other in decomposed[1:]:
        result *= other.scale
    return result


class Dimension(symbolic.Expression):
    """An symbolic expression representing a physical dimension."""

    def __init__(self, arg: symbolic.Expressable) -> None:
        super().__init__(arg)
        self._quantities = {}

    def quantities(self, system: str) -> typing.Set[str]:
        """The quantities with this dimension in `system`."""
        if system in self._quantities:
            return self._quantities[system]
        canonical = CANONICAL['dimensions'][system]
        found = {k for k, v in canonical.items() if v == self}
        self._quantities[system] = found
        return found


def dimension_factory(arg: symbolic.Expressable):
    """Factory function for metric dimensions."""
    return Dimension(symbolic.expression(arg))


class Dimensions(typing.Mapping):
    """A collection of symbolic expressions of metric dimensions."""

    def __init__(
        self,
        common: symbolic.Expressable=None,
        **systems: symbolic.Expressable
    ) -> None:
        """Initialize from expressable quantities.
        
        Parameters
        ----------
        common : string or iterable or `~symbolic.Expression`
            The dimension to associate with all metric systems.

        **systems
            Zero or more key-value pairs in which the key is the name of a known
            metric system and the value is an object that can instantiate the
            `~symbolic.Expression` class. If present, each value will override
            `common` for the corresponding metric system.
        """
        self._objects = self._init_from(common, **systems)

    def _init_from(
        self,
        common,
        **systems
    ) -> typing.Dict[str, typing.Optional[Dimension]]:
        """Create dimension objects from arguments."""
        created = dict.fromkeys(_reference.SYSTEMS)
        default = Dimension(common) if common else None
        updates = {
            system: Dimension(expressable) if expressable else default
            for system, expressable in systems.items()
        }
        created.update(updates)
        if any(created.values()):
            return created
        raise TypeError(
            f"Can't instantiate {self.__class__!r}"
            f" from {common!r} and {systems!r}"
        ) from None

    def __len__(self) -> int:
        return len(self._objects)

    def __iter__(self) -> typing.Iterator:
        return iter(self._objects)

    def __getitem__(self, __k):
        key = str(__k).lower()
        if key in self._objects:
            return self._objects[key]
        raise KeyError(f"No dimension for {__k!r}") from None

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"{self.__class__.__qualname__}({self})"

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return ', '.join(
            f"{k!r}: {str(v) if v else v!r}"
            for k, v in self._objects.items()
        )


class _Property(collections.abc.Mapping):
    """All definitions of a single metric property."""

    _instances = {}
    _supported = (
        'dimensions',
        'units',
    )

    key: str=None
    _cache: dict=None

    def __new__(
        cls: typing.Type[typing.Self],
        arg: typing.Union[str, typing.Self],
    ) -> typing.Self:
        """Create a new instance or return an existing one.
        
        Parameters
        ----------
        arg : string or instance
            A string representing the metric property to create, or an existing
            instance of this class.
        """
        if isinstance(arg, cls):
            return arg
        key = str(arg)
        if key not in cls._supported:
            raise ValueError(f"Unsupported property: {key}") from None
        if available := cls._instances.get(key):
            return available
        self = super().__new__(cls)
        self.key = key
        self._cache = {}
        cls._instances[key] = self
        return self

    def system(self, system: str):
        """Get all definitions of this property for `system`."""
        return {k: v[system] for k, v in self.items()}

    LEN = len(_reference.QUANTITIES) # No need to compute every time.

    def __len__(self) -> int:
        """The number of defined quantities. Called for len(self)."""
        return self.LEN

    def __iter__(self) -> typing.Iterator[str]:
        """Iterate over names of defined quantities. Called for iter(self)."""
        return iter(_reference.QUANTITIES)

    def __getitem__(self, quantity: str) -> typing.Dict[str, str]:
        """Create or retrieve a named property."""
        if quantity in self._cache:
            return self._cache[quantity]
        if new := self._get_property(quantity):
            self._cache[quantity] = new
            return new
        raise KeyError(f"Unknown quantity {quantity}") from None

    def _get_property(self, quantity: str) -> typing.Dict[str, str]:
        """Get this property for a defined quantity.
        
        This method will search for `quantity` in the module-level collection of
        defined quantities. If it doesn't find an entry, it will attempt to
        parse `quantity` into known quantities. If it finds a `dict` entry, it
        will attempt to extract the values corresponding to this property's
        `key` attribute (i.e., 'units' or 'dimensions'). If it finds a `str`
        entry, it will attempt to create the equivalent `dict` by symbolically
        evaluating the terms in the entry.
        """
        if quantity not in _reference.QUANTITIES:
            return self._parse(quantity)
        q = _reference.QUANTITIES[quantity]
        if isinstance(q, dict):
            return q.get(self.key, {})
        if not isinstance(q, str):
            raise TypeError(f"Expected {quantity} to be a string") from None
        return self._parse(q)

    def _parse(self, string: str):
        """Parse a string representing a compound quantity."""
        for k in _reference.QUANTITIES:
            string = string.replace(k, k.replace(' ', '_'))
        parts = [
            self._expand(term)
            for term in symbolic.expression(string)
        ]
        keys = {key for part in parts for key in part.keys()}
        merged = {key: [] for key in keys}
        for part in parts:
            for key, value in part.items():
                merged[key].append(value)
        return {
            k: str(symbolic.expression(v))
            for k, v in merged.items()
        }

    _operand = symbolic.OperandFactory()
    def _expand(self, term: symbolic.Term):
        """Create a `dict` of operands from this term."""
        return {
            k: self._operand.create(v, term.exponent)
            for k, v in self[term.base.replace('_', ' ')].items()
        }

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"{self.__class__.__qualname__}({self})"

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return self.key


# NOTE: Defining mappings from unit or dimension to quantity is a bad idea
# because some quantities have the same unit or dimension in a given system.
# This makes the mapping ill-defined. Python dictionaries simply use the latest
# entry for a repeated key, which means some quantities would overwrite others.
# The following objects rely on mappings from quantity to unit or dimension,
# which are always well defined.


DIMENSIONS = _Property('dimensions')
"""All defined metric dimensions.

This mapping is keyed by physical quantity followed by metric system.
"""


UNITS = _Property('units')
"""All defined metric units.

This mapping is keyed by physical quantity followed by metric system.
"""


CANONICAL = {
    k: {
        system: _Property(k).system(system) for system in ('mks', 'cgs')
    } for k in ('dimensions', 'units')
}
"""Canonical metric properties in each known metric system.

This mapping is keyed by {'dimensions', 'units'}, followed by metric system, and
finally by physical quantity.
"""


MKS = {k: CANONICAL[k]['mks'].copy() for k in ('dimensions', 'units')}
"""All defined units in the MKS system, keyed by physical quantity."""


CGS = {k: CANONICAL[k]['cgs'].copy() for k in ('dimensions', 'units')}
"""All defined units in the CGS system, keyed by physical quantity."""


_EQUIVALENT = {
    p: {
        k: {
            s: symbolic.expression(x)
            for s, x in v.items()
        } for k, v in dict(_Property(p)).items()
    } for p in ('dimensions', 'units')
}

EQUIVALENT = aliasedkeys.Mapping(_EQUIVALENT)


class Prefix(typing.NamedTuple):
    """Metadata for a metric order-of-magnitude prefix."""

    symbol: str
    name: str
    factor: float


class BaseUnit(typing.NamedTuple):
    """Metadata for a named unit without metric prefix."""

    symbol: str
    name: str
    quantity: str


Instance = typing.TypeVar('Instance', bound='NamedUnit')


class _NamedUnitMeta(abc.ABCMeta):
    """Internal metaclass for `~metric.NamedUnit`.
    
    This class exists to create singleton instances of `~metric.NamedUnit`
    without needing to overload `__new__` on that class or its base class(es).
    """

    _instances = aliasedkeys.MutableMapping()
    _attributes = {}

    def __call__(
        cls,
        arg: typing.Union[Instance, symbolic.Expressable],
    ) -> Instance:
        """Create a new instance or return an existing one."""
        if isinstance(arg, cls):
            # If the argument is already an instance, return it.
            return arg
        string = str(arg)
        if available := cls._instances.get(string):
            # If the argument maps to an existing unit, return that unit.
            return available
        # First time through: identify the base unit and prefix.
        magnitude, reference = identify(string)
        # Store the information to initialize a new instance.
        name = f"{magnitude.name}{reference.name}"
        symbol = f"{magnitude.symbol}{reference.symbol}"
        cls._attributes[string] = {
            'prefix': magnitude,
            'base': reference,
            'name': name,
            'symbol': symbol,
            'scale': magnitude.factor,
            'quantity': reference.quantity,
        }
        # Create the new instance. This will ultimately pass control to
        # `NamedUnit.__init__`, which will initialize the newly instantiated
        # instance with the stored attributes corresponding to `str(arg)`.
        instance = super().__call__(arg)
        cls._instances[(name, symbol)] = instance
        return instance


def identify(string: str):
    """Determine the magnitude and reference unit, if possible.
    
    Parameters
    ----------
    string : str
        A string representing a metric unit.

    Returns
    -------
    tuple
        A 2-tuple in which the first element is a `~metric.Prefix` representing
        the order-of-magnitude of the given unit and the second element is a
        `~metric.BaseUnit` representing the unscaled (i.e., order-unity) metric
        unit.

    Examples
    --------
    >>> mag, ref = identify('km')
    >>> mag
    Prefix(symbol='k', name='kilo', factor=1000.0)
    >>> ref
    BaseUnit(symbol='m', name='meter', quantity='length')
    """
    try:
        unit = _reference.NAMED_UNITS[string]
    except KeyError as err:
        raise UnitParsingError(string) from err
    magnitude = Prefix(**unit['prefix'])
    reference = BaseUnit(**unit['base'])
    return magnitude, reference


class Reduction:
    """The components of a reduced unit expression."""

    def __init__(
        self,
        terms: symbolic.Expressable,
        scale: float=1.0,
        system: str=None,
    ) -> None:
        self._expression = scale * symbolic.expression(terms)
        self.system = system
        self.scale = scale
        self._units = None

    @property
    def units(self) -> typing.List[symbolic.Term]:
        """The unit terms in this reduction."""
        if self._units is None:
            self._units = [
                unit for unit in self._expression.terms
                if unit.base != '1'
            ]
        return self._units

    def __mul__(self, other):
        """Called for self * other."""
        if not isinstance(other, numbers.Real):
            return NotImplemented
        scale = self.scale * other
        terms = list(scale * self._expression)
        return type(self)(terms, scale=scale, system=self.system)

    __rmul__ = __mul__
    """Called for other * self."""

    def __truediv__(self, other):
        """Called for self / other."""
        if not isinstance(other, numbers.Real):
            return NotImplemented
        scale = self.scale / other
        terms = list(scale * self._expression)
        return type(self)(terms, scale=scale, system=self.system)

    def __pow__(self, other):
        """Called for self ** other."""
        if not isinstance(other, numbers.Real):
            return NotImplemented
        scale = self.scale ** other
        terms = list(scale * self._expression**other)
        return type(self)(terms, scale=scale, system=self.system)

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"{self.__class__.__qualname__}({self})"

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return f"{self._expression} [{self.system!r}]"


class NamedUnit(metaclass=_NamedUnitMeta):
    """A single named unit and corresponding metadata."""

    @typing.overload
    def __init__(
        self: Instance,
        unit: str,
    ) -> Instance:
        """Create a new instance or return an existing one.
        
        Parameters
        ----------
        unit : string
            A string representing the metric unit to create.
        """

    @typing.overload
    def __init__(
        self: Instance,
        instance: Instance,
    ) -> Instance:
        """Create a new instance or return an existing one.
        
        Parameters
        ----------
        instance : `~metric.NamedUnit`
            An existing instance of this class.
        """

    def __init__(self, arg) -> None:
        self._parsed = self.__class__._attributes[str(arg)]
        self._prefix = None
        self._base = None
        self._name = None
        self._symbol = None
        self._scale = None
        self._quantity = None
        self._systems = None
        self._dimensions = None
        self._decomposed = None
        self._norm = None
        self._reductions = dict.fromkeys(_reference.SYSTEMS)

    @property
    def norm(self):
        """The equivalent unit, represented in base units of `system`.
        
        Notes
        -----
        This property returns a copy of the original `dict` of normalized units
        in order to prevent modifying singleton instances.
        """
        if self._norm is None:
            self._norm = {
                system: type(self)(UNITS[self.quantity][system])
                for system in _reference.SYSTEMS
            }
        return self._norm.copy()

    @property
    def prefix(self) -> Prefix:
        """The order of magnitide of this unit's metric prefix."""
        if self._prefix is None:
            self._prefix = self._parsed["prefix"]
        return self._prefix

    @property
    def base(self) -> BaseUnit:
        """The reference unit without metric prefix."""
        if self._base is None:
            self._base = self._parsed["base"]
        return self._base

    @property
    def name(self) -> str:
        """The full name of this unit."""
        if self._name is None:
            self._name = self._parsed["name"]
        return self._name

    @property
    def symbol(self) -> str:
        """The abbreviated symbol for this unit."""
        if self._symbol is None:
            self._symbol = self._parsed["symbol"]
        return self._symbol

    @property
    def scale(self) -> float:
        """The metric scale factor of this unit."""
        if self._scale is None:
            self._scale = self._parsed["scale"]
        return self._scale

    @property
    def quantity(self) -> str:
        """The physical quantity of this unit."""
        if self._quantity is None:
            self._quantity = self._parsed["quantity"]
        return self._quantity

    @property
    def systems(self):
        """The metric systems that use this unit.
        
        This property uses the criteria described in
        `~metric.NamedUnit.is_allowed_in` to build the collection of metric
        systems, most notably that named units not defined in any metric system
        are allowed in all metric systems.
        """
        if self._systems is None:
            modes = {
                k: []
                for k in {'allowed', 'defined', 'fundamental'}
            }
            for system in _reference.SYSTEMS:
                if self.is_fundamental_in(system):
                    modes['fundamental'].append(system)
                if self.is_defined_in(system):
                    modes['defined'].append(system)
                if self.is_allowed_in(system):
                    modes['allowed'].append(system)
            self._systems = {k: tuple(v) for k, v in modes.items()}
        return self._systems.copy()

    @property
    def decomposed(self):
        """The representation of this unit in base units, if possible."""
        if self._decomposed is None:
            with contextlib.suppress(StopIteration):
                system = next(
                    system for system in _reference.SYSTEMS
                    if self.is_fundamental_in(system)
                )
                self._decomposed = self._decompose(system)
        return self._decomposed

    def _decompose(self, system: typing.Literal['mks', 'cgs']):
        """Internal logic for `NamedUnit.decomposed`."""
        if not self.is_defined_in(system):
            # If this unit is not defined in this metric system, we can't
            # decompose it.
            return
        dimension = self.dimensions[system]
        expression = symbolic.expression(dimension)
        if len(dimension) == 1:
            # If this unit's dimension is irreducible, there's no point in going
            # through all the decomposition logic.
            return [symbolic.term(self.symbol)]
        quantities = [
            find(_reference.BASE_QUANTITIES, term.base, unique=True)['name']
            for term in expression
        ]
        units = [
            _reference.QUANTITIES[quantity]['units'][system]
            for quantity in quantities
        ]
        return [
            symbolic.term(base=unit, exponent=term.exponent)
            for unit, term in zip(units, expression)
        ]

    def reduce(self, system: str=None) -> typing.Optional[Reduction]:
        """Convert this unit to base units of `system`, if possible."""
        s = self._resolve_system(system)
        if self._reductions[s]:
            return self._reductions[s]
        result = self._reduce(s)
        self._reductions[s] = result
        return result

    def _resolve_system(self, system: typing.Optional[str]):
        """Determine the appropriate metric system to use, if possible."""
        if isinstance(system, str) and system.lower() in _reference.SYSTEMS:
            # trivial case
            return system.lower()
        systems = [s for s in _reference.SYSTEMS if self.is_fundamental_in(s)]
        if len(systems) == 1:
            # use canonical system if possible
            return systems[0]
        if self.dimensions['mks'] == self.dimensions['cgs']:
            # system-independent: use mks by default
            return 'mks'
        if self.dimensions['mks'] is None:
            # only defined in cgs
            return 'cgs'
        if self.dimensions['cgs'] is None:
            # only defined in mks
            return 'mks'
        # system-dependent but we don't know the system
        raise SystemAmbiguityError(str(self))

    def _reduce(self, system: typing.Literal['mks', 'cgs']):
        """Internal logic for `~NamedUnit.reduce`."""
        if not self.is_defined_in(system):
            # If this unit is not defined in this metric system, we can't reduce
            # it.
            return
        dimension = self.dimensions[system]
        expression = symbolic.expression(dimension)
        if len(expression) == 1:
            # If this unit's dimension is irreducible, there's no point in going
            # through all the reduction logic.
            canonical = CANONICAL['units'][system][self.quantity]
            if self.symbol == canonical:
                # If this is the canonical unit for its quantity in `system`,
                # return it with a scale of unity.
                return Reduction(
                    [symbolic.term(self.symbol)],
                    system=system,
                )
            # If not, return the canonical unit with the appropriate scale
            # factor.
            return Reduction(
                [symbolic.term(canonical)],
                scale=(canonical << self),
                system=system,
            )
        quantities = [
            find(_reference.BASE_QUANTITIES, term.base, unique=True)['name']
            for term in expression
        ]
        units = [
            _reference.QUANTITIES[quantity]['units'][system]
            for quantity in quantities
        ]
        terms = [
            symbolic.term(base=unit, exponent=term.exponent)
            for unit, term in zip(units, expression)
        ]
        return Reduction(terms, scale=self.scale, system=system)

    @property
    def dimensions(self) -> typing.Dict[str, typing.Optional[str]]:
        """The physical dimension of this unit in each metric system.
        
        Notes
        -----
        This property returns a copy of an internal `dict` in order to prevent
        accidentally changing the instance dimensions through an otherwise valid
        `dict` operation. Such changes are irreversible since each
        `~metric.NamedUnit` instance is a singleton.
        """
        if self._dimensions is None:
            systems = {system for system in self.systems['defined']}
            self._dimensions = {
                k: symbolic.expression(v) or None
                for k, v in self._get_dimensions(systems).items()
            }
        return self._dimensions.copy()

    def _get_dimensions(self, systems: typing.Set[str]):
        """Helper for computing dimensions of this named unit.
        
        Notes
        -----
        This method requires the full set of applicable metric systems (rather
        than one system at a time) because it will return all available
        dimensions if either 1) there are no applicable systems, or 2) the set
        of applicable systems is equal to the set of all known systems.
        """
        dimensions = DIMENSIONS[self.quantity]
        if not systems or (systems == _reference.SYSTEMS):
            return dimensions.copy()
        base = dict.fromkeys(_reference.SYSTEMS)
        if len(systems) == 1:
            system = systems.pop()
            base[system] = dimensions[system]
            return base
        raise SystemAmbiguityError

    def is_allowed_in(self, system: typing.Literal['mks', 'cgs']):
        """True if this named unit inter-operates with units in `system`.
        
        A named unit is allowed in some or all metric systems, but never none.
        The reason for this is that a named unit that is not defined in any
        metric system is effectively independent of all metric systems, so
        attempting to restrict its use to a subset of metric systems is
        fruitless.

        See Also
        --------
        `~is_defined_in`
            True if the given metric system formally contains this named unit.

        `~is_fundamental_in`
            True if this named unit is the fundamental unit for its dimension in
            the given metric system.
        """
        systems = {
            s for s in _reference.SYSTEMS if self.is_defined_in(s)
        } or _reference.SYSTEMS
        return system in systems

    def is_defined_in(self, system: typing.Literal['mks', 'cgs']):
        """True if this named unit is defined in `system`."""
        if self.is_fundamental_in(system):
            return True
        canonical = CANONICAL['units'][system][self.quantity]
        with contextlib.suppress(UnitParsingError):
            reference = type(self)(canonical)
            if self.base == reference.base:
                return True
        return False

    def is_fundamental_in(self, system: typing.Literal['mks', 'cgs']):
        """True if this named unit is the canonical unit in `system`."""
        canonical = CANONICAL['units'][system][self.quantity]
        keys = (self.symbol, self.name)
        for key in keys:
            if key == canonical:
                return True
        return False

    def __eq__(self, other) -> bool:
        """True if two representations have equal magnitude and base unit."""
        that = type(self)(other)
        same_magnitude = (self.prefix == that.prefix)
        same_reference = (self.base == that.base)
        return same_magnitude and same_reference

    def __hash__(self) -> int:
        """Called for hash(self). Supports use as mapping key."""
        return hash((self.base, self.prefix))

    def __rshift__(self, other):
        """Compute the conversion factor from this unit to `other`.

        This operation computes the numerical factor necessary to convert a
        quantity in terms of this unit to a quantity in terms of `other`,
        provided both units have the same base unit.

        Examples
        --------
        The result of this operation is equal to the result of `~metric.ratio`:

        >>> metric.ratio('cm', 'm')
        0.01
        >>> metric.NamedUnit('cm') >> 'm'
        0.01
        """
        if isinstance(other, (NamedUnit, str)):
            return ratio(self, other)
        return NotImplemented

    def __rrshift__(self, other):
        """Reflected version of `__rshift__`."""
        if isinstance(other, str):
            return ratio(other, self)
        return NotImplemented

    def __lshift__(self, other):
        """Compute the conversion factor from `other` to this unit.
        
        This operation computes the numerical factor necessary to convert a
        quantity in terms of `other` to a quantity in terms of this unit,
        provided both units have the same base unit.

        Examples
        --------
        The result of this operation is equal to the inverse of the result of
        `~metric.ratio`:

        >>> metric.ratio('cm', 'm')
        0.01
        >>> metric.NamedUnit('cm') << 'm'
        100.0
        """
        if isinstance(other, (NamedUnit, str)):
            return ratio(other, self)
        return NotImplemented

    def __rlshift__(self, other):
        """Reflected version of `__lshift__`."""
        if isinstance(other, str):
            return ratio(self, other)
        return NotImplemented

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"{self.__class__.__qualname__}({self})"

    def __str__(self) -> str:
        """A simplified representation of this unit."""
        return f"'{self.name} | {self.symbol}'"


M = typing.TypeVar('M', bound=typing.Mapping)


def find(
    entries: typing.Collection[M],
    value: T,
    unique: bool=False,
) -> typing.Union[M, typing.List[M]]:
    """Find entries with the given value."""
    found = [entry for entry in entries if value in entry.values()]
    if not found:
        raise ValueError(f"No entries containing {value!r}")
    if not unique:
        return found
    if len(found) > 1:
        raise ValueError(f"No unique entry containing {value!r}")
    return found[0]


def ratio(
    this: typing.Union[str, NamedUnit],
    that: typing.Union[str, NamedUnit],
) -> float:
    """Compute the magnitude of `this` relative to `that`.

    Parameters
    ----------
    this : string or `~metric.NamedUnit`
        The reference unit.

    that : string or `~metric.NamedUnit`
        The unit to compare to `this`. Must have the same base unit as `this`.

    Examples
    --------
    The following are all equivalent to the fact that a meter represents 100
    centimeters:

        >>> ratio('meter', 'centimeter')
        100.0
        >>> ratio('centimeter', 'meter')
        0.01
        >>> ratio('m', 'cm')
        100.0
        >>> ratio('cm', 'm')
        0.01

    Attempting this operation between two units with different base units will
    raise an exception,

        >>> ratio('m', 's')
        <raises ValueError>

    even if they represent the same quantity.

        >>> ratio('m', 'au')
        <raises ValueError>

    Therefore, this function is not intended for unit conversion.
    """
    u = [NamedUnit(i) if isinstance(i, str) else i for i in (this, that)]
    if not all(isinstance(i, NamedUnit) for i in u):
        raise TypeError(
            f"Each unit must be an instance of {str!r} or {NamedUnit!r}"
        ) from None
    u0, u1 = u
    if u1 == u0:
        return 1.0
    if u1.base != u0.base:
        units = ' to '.join(f"{i.symbol!r} ({i.name})" for i in u)
        raise ValueError(f"Can't compare {units}") from None
    return u0.scale / u1.scale


def get_quantity(unit: symbolic.Expression):
    """Compute the physical quantity of the given unit."""
    terms = [
        symbolic.term(
            NamedUnit(term.base).quantity.replace(' ', '_')
        ) ** term.exponent
        for term in unit
    ]
    return symbolic.expression(terms)


class Conversion:
    """The result of a unit conversion."""

    def __init__(self, u0: str, u1: str, factor: float=1.0) -> None:
        self._u0 = u0
        self._u1 = u1
        self._factor = factor

    def __rtruediv__(self, other):
        """Called to convert this conversion to its inverse."""
        if other != 1:
            clsname = self.__class__.__qualname__.lower()
            raise ValueError(
                f"Cannot compute {other} / <{clsname}>"
                f"; use {other} / float(<{clsname}>)"
            ) from None
        return Conversion(self.u1, self.u0, 1.0 / self._factor)

    def __float__(self) -> float:
        """Reduce this instance to its numerical factor via float(self)."""
        if bool(self):
            return self._factor
        raise TypeError("Conversion is undefined") from None

    def __bool__(self):
        """True if this conversion exists."""
        return self._factor is not None

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        attrstr = f"{self.u0!r}, {self.u1!r}, {self._factor}"
        return f"{self.__class__.__qualname__}({attrstr})"

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return f"({self.u0!r} -> {self.u1!r}) == {float(self)}"

    @property
    def u0(self):
        """The unit from which to convert."""
        return self._u0

    @property
    def u1(self):
        """The unit to which to convert."""
        return self._u1


_CONVERSIONS = {}
"""Internal collection of computed unit conversions.

This module-level collection is a mapping from unit pairs to their corresponding
`~Conversion` instances. It exists to preclude the need to repeatedly compute
a given conversion.
"""

def conversion_factory(u0: str, u1: str) -> Conversion:
    """Compute the conversion from `u0` to `u1`.

    Parameters
    ----------
    u0 : string
        The unit from which to convert.

    u1 : string
        The unit to which to convert.

    Returns
    -------
    `~Conversion`

    Raises
    ------
    `~UnitConversionError`
        All attempts to convert `u0` to `u1` failed.

    Notes
    -----
    - This function will store both the successful conversion and its inverse in
      an internal module-level `dict` so that subsequent requests for the same
      conversion or its inverse do not trigger redundant computation.
    """
    if available := _CONVERSIONS.get((u0, u1)):
        return available
    c = _convert(u0, u1)
    _CONVERSIONS[(u0, u1)] = c
    _CONVERSIONS[(u1, u0)] = 1 / c
    return c


def _convert(u0: str, u1: str):
    """Attempt to compute the conversion from `u0` to `u1`."""
    converters = (
        _convert_as_strings,
        _convert_as_expressions,
    )
    for converter in converters:
        if factor := converter(u0, u1):
            return Conversion(u0, u1, factor)
    raise UnitConversionError(
        f"Cannot convert {u0!r} to {u1!r}"
    ) from None


def _convert_as_strings(u0: str, u1: str) -> typing.Optional[float]:
    """Attempt to convert `u0` to `u1` as strings."""
    checked = []
    if factor := _recursive_conversion(checked, u0, u1):
        return factor


def _recursive_conversion(
    checked: typing.List[str],
    u0: str,
    u1: str,
    scale: float=1.0,
) -> typing.Optional[float]:
    """Attempt to convert `u0` to `u1` as strings."""
    if u0 not in checked:
        checked += [u0]
        if factor := _standard_conversion(u0, u1, scale=scale):
            return factor
        conversions = _reference.CONVERSIONS.get_adjacencies(u0).items()
        for unit, weight in conversions:
            if value := _recursive_conversion(checked, unit, u1, scale=scale):
                return weight * value


def _standard_conversion(
    u0: str,
    u1: str,
    scale: float=1.0,
) -> typing.Optional[float]:
    """Attempt to convert `u0` to `u1` non-recursively.

    This function will attempt to compute the appropriate numerical
    conversion factor via one of the following strategies

    - looking up a defined conversion
    - computing the ratio of metric scale factors
    - transitively scaling through an intermediate conversion
    """
    if factor := _simple_conversion(u0, u1, scale=scale):
        return factor
    if factor := _rescale_conversion(u0, u1, scale=scale):
        return factor


def _convert_as_expressions(u0: str, u1: str):
    """Convert complex unit expressions term-by-term."""
    e0, e1 = (symbolic.expression(unit) for unit in (u0, u1))
    if e0 == e1:
        return 1.0
    terms = [
        term for term in (e0 / e1).terms
        if term.base not in _reference.UNITY
    ]
    if factor := _resolve_terms(terms):
        return factor
    if factor := _convert_by_dimensions(terms):
        return factor


def _simple_conversion(u0: str, u1: str, scale: float=1.0):
    """Attempt to compute a simple conversion from `u0` to `u1`.
    
    This method will attempt the following conversions, in the order listed:
    
    * the identity conversion (i.e., `u0 == u1`);
    * a defined conversion from `u0` to `u1`;
    * the metric ratio of `u1` to `u0` (e.g., `'km'` to `'m'`);

    If it does not succeed, it will return `None` in order to allow other
    methods an opportunity to convert `u0` to `u1`.
    """
    if u0 == u1:
        return scale
    if found := _search(u0, u1):
        return scale * found
    try:
        ratio = NamedUnit(u0) >> NamedUnit(u1)
    except (ValueError, UnitParsingError):
        return
    return scale * ratio


def _rescale_conversion(u0: str, u1: str, scale: float=1.0):
    """Attempt to convert `u0` to `u1` via an intermediate unit.

    This method will attempt the following conversions, in the order listed:

    - a defined conversion to `u1` from a unit that has the same base unit as,
      but different metric scale than, `u0` (see `~Conversion._rescale`);
    - the numerical inverse of the above process applied to the conversion from
      `u1` to `u0`;

    If it does not succeed, it will return `None` in order to allow other
    methods an opportunity to convert `u0` to `u1`.
    
    Notes
    -----
    It is possible for `u0` or `u1` to individually be in `CONVERSIONS.nodes`
    even when `(u0, u1)` is not in `CONVERSIONS`. For example, there are nodes
    for both 'min' (minute) and 'd' (day), each with a conversion to 's'
    (second), but there is no direct conversion from 'min' to 'd'.
    """
    if u0 not in _reference.CONVERSIONS.nodes:
        if computed := _rescale(u0, u1):
            return scale * computed
    if u1 not in _reference.CONVERSIONS.nodes:
        if computed := _rescale(u1, u0):
            return scale / computed


def _rescale(u0: str, u1: str):
    """Compute a new conversion after rescaling `u0`.
    
    This method will look for a unit, `ux`, in `~metric.CONVERSIONS` that has
    the same base unit as `u0`. If it finds one, it will attempt to convert `ux`
    to `u1`, and finally multiply the result by the relative magnitude of `u0`
    to `ux`. In other words, it attempts to compute `(u0 << ux) * (ux -> u1)` in
    place of `(u0 -> u1)`.

    For example, if the direct conversion from megajoules to ergs is not
    available but the direct conversions from megajoules to joules and from
    joule to ergs are, this function will attempt to compute `('MJ' -> 'erg') =
    ('MJ' -> 'J') * ('J' -> 'erg')`.
    """
    if not u0 in _reference.NAMED_UNITS:
        return
    n0 = NamedUnit(u0)
    for ux in _reference.CONVERSIONS.nodes:
        if ux in _reference.NAMED_UNITS:
            nx = NamedUnit(ux)
            if nx.base == n0.base:
                if found := _search(ux, u1):
                    return (nx << n0) * found


def _search(u0: str, u1: str):
    """Search the defined conversions.
    
    This method will first search for a defined conversion from `u0` to `u1`. If
    it can't find one, it will search for a defined conversion from an alias of
    `u0` to an alias of `u1`. See `~_get_aliases_of` for more information.
    """
    if (u0, u1) in _reference.CONVERSIONS:
        return _reference.CONVERSIONS.get_weight(u0, u1)
    starts = _get_aliases_of(u0)
    ends = _get_aliases_of(u1)
    for ux in starts:
        for uy in ends:
            if (ux, uy) in _reference.CONVERSIONS:
                return _reference.CONVERSIONS.get_weight(ux, uy)


def _get_aliases_of(unit: str):
    """Build a list of possible variations of this unit string.
    
    The aliases of `unit` comprise the given string, as well as the canonical
    name and symbol of the corresponding named unit, if one exists.
    """
    built = [unit]
    if unit in _reference.NAMED_UNITS:
        known = NamedUnit(unit)
        built.extend([known.symbol, known.name])
    return built


def _convert_by_dimensions(terms: typing.List[symbolic.Term]):
    """Attempt to compute a conversion via unit dimensions."""
    factor = 1.0
    decomposed = []
    for term in terms:
        if decomposition := _convert_by_decomposition(term):
            scale, units = decomposition
            decomposed.extend(units)
            factor *= scale
    if symbolic.expression(decomposed) == '1':
        return factor
    cancelled = _cancel_terms(decomposed)
    reduced = _reduce_terms(cancelled)
    if result := _resolve_terms(reduced):
        return factor * result


def _convert_by_decomposition(term: symbolic.Term):
    """Convert a unit into fundamental units, if possible."""
    exponent = term.exponent
    if reduction := _reduce_unit(term.base, exponent):
        return 1.0, reduction
    for system in _reference.SYSTEMS:
        norm = NamedUnit(term.base).norm[system]
        if reduction := _reduce_unit(norm.symbol, exponent):
            if factor := _convert_as_strings(term.base, norm.symbol):
                return factor**exponent, reduction


def _reduce_unit(base: str, exponent: numbers.Real):
    """Compute the equivalent expression in fundamental units, if possible."""
    if reduction := NamedUnit(base).reduce():
        return [
            symbolic.term(
                coefficient=reduction.scale**exponent,
                base=this.base,
                exponent=exponent*this.exponent,
            )
            for this in reduction.units
        ]


def _cancel_terms(terms: typing.List[symbolic.Term]):
    """Cancel out terms with equal magnitude and opposite exponent."""
    # TODO: Reduce redundancy with `_resolve_terms`.
    matched = []
    unmatched = terms.copy()
    for target in terms:
        if target not in matched:
            inverse_powers = [
                term for term in terms
                if term != target and term.exponent == -target.exponent
            ]
            for term in inverse_powers:
                if term.base == target.base:
                    for this in (target, term):
                        matched.append(this)
                        unmatched.remove(this)
    return unmatched


def _resolve_terms(terms: typing.List[symbolic.Term]):
    """Compute ratios of terms with comparable exponents, if possible."""
    # TODO: Reduce redundancy with `_cancel_terms`.
    if len(terms) <= 1:
        # We require at least two terms for a ratio.
        return
    factor = 1.0
    matched = []
    unmatched = terms.copy()
    for target in terms:
        if target not in matched:
            if match := _match_exponents(target, unmatched):
                value, term = match
                if term != target:
                    for this in (target, term):
                        matched.append(this)
                        unmatched.remove(this)
                else:
                    matched.append(target)
                    unmatched.remove(target)
                factor *= value
    if not unmatched:
        return factor


def _reduce_terms(terms: typing.List[symbolic.Term]):
    """Partially or fully cancel terms with equal bases."""
    return symbolic.expression(terms).terms


def _match_exponents(
    target: symbolic.Term,
    terms: typing.Iterable[symbolic.Term],
) -> typing.Optional[typing.Tuple[float, symbolic.Term]]:
    """Attempt to convert `target` to a term in `terms` by exponent.

    This function first checks whether `target` is a dimensionless reference
    unit and, if so, returns a conversion factor of 1.0 along with `target`. If
    that check fails, this function compares `target` to each term in `terms`
    that has an exponent with the same magnitude and opposite sign. If it can
    convert the base unit of `target` to the base unit of a term in `terms`, it
    will return the corresponding conversion factor and the matching term.
    """
    u0 = target.base
    dimensions = NamedUnit(u0).dimensions.values()
    if u0 in _reference.NAMED_UNITS and all(d == '1' for d in dimensions):
        return 1.0, target
    exponent = target.exponent
    inverse_powers = [
        term for term in terms
        if term != target and term.exponent == -exponent
    ]
    for term in inverse_powers:
        u1 = term.base
        if conversion := _convert_as_strings(u0, u1):
            return conversion ** exponent, term


class Converter:
    """Unit-conversion handler for a known physical quantity."""

    def __init__(
        self,
        unit: str,
        quantity: str,
        substitutions: typing.Dict[str, str],
    ) -> None:
        self._unit = substitutions.get(unit) or unit
        self._quantity = quantity
        self._substitutions = substitutions

    def to(self, target: str):
        """Compute the conversion from the current unit to `target`.
        
        Parameters
        ----------
        target : string
            The unit or metric system to which to convert the current unit.

        Returns
        -------
        `~Conversion`

        Notes
        -----
        This method proceeds as follows:
        
        1. Substitute the appropriate unit if `target` is a metric system.
        1. Immediately return a trivial conversion object (i.e., with a `factor`
           of 1.0) if the target unit and the current unit are equal.
        1. Attempt a brute-force conversion via the `~factory` function and, if
           successful, return the resulting conversion object.
        1. Raise an exception to alert the caller that the conversion is
           undefined.
        """
        unit = self._substitutions.get(target) or target
        if self.unit == unit:
            return Conversion(self.unit, unit)
        if result := conversion_factory(self.unit, unit):
            return result
        raise ValueError(
            f"Unknown conversion from {self.unit!r} to {unit!r}."
        ) from None

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"{self.__class__.__qualname__}({self})"

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return f"{self.unit!r} [{self.quantity!r}]"

    @property
    def unit(self):
        """The unit from which to convert."""
        return self._unit

    @property
    def quantity(self):
        """The physical quantity of `unit`."""
        return self._quantity


_converters = {}
"""Internal collection of unit-conversion handlers.

This module-level collection is a mapping from unit-quantity pairs to their
correpsonding `~Converter` instances. It exists to preclude the existence of
redundant unit-conversion handlers.
"""

def converter_factory(unit: str, quantity: str) -> Converter:
    """Create a new unit-conversion handler.

    Parameters
    ----------
    unit : string
        The unit to be converted.

    quantity : string
        The physical quantity of `unit`.

    Returns
    -------
    `~Converter`
        The unit-conversion handler corresponding to `unit` and `quantity`.

    Raises
    ------
    `ValueError`
        The given quantity is not a known physical quantity.
    """
    if available := _converters.get((unit, quantity)):
        return available
    substitutions = _get_substitutions(quantity)
    c = Converter(unit, quantity, substitutions)
    _converters[(unit, quantity)] = c
    return c


def _get_substitutions(quantity: str):
    """Get appropriate metric-system unit substitutions for `~converter`.
    
    This function exists to handle edge cases, such as treating atomic mass
    number quantities as mass quantities when converting to the equivalent unit
    in a given metric system.
    
    As a specific example, suppose you have a mass-like quantity with unit
    'nuc'. Since the metric quantity of 'nuc' is 'mass number', not 'mass', the
    default unit substitution would be '1' for both 'mks' and 'cgs'. However,
    intuition suggests that converting a mass-like quantity to 'mks' or 'cgs'
    should produce a quantity with unit 'kg' or 'g', respectively. This function
    therefore handles the special case of 'mass number' by replacing the default
    substitutions with the substitutions for 'mass'.
    """
    if quantity.lower() in {'mass number', 'mass_number'}:
        return UNITS['mass']
    try:
        substitutions = UNITS[quantity]
    except KeyError as err:
        raise ValueError(
            f"Unknown quantity {quantity!r}"
        ) from err
    return substitutions


def convert(source: str, target: str, quantity: str=None):
    """Convert `source` to `target`."""
    if not isinstance(source, str):
        raise TypeError(
            f"Argument to source must be a string, not {type(source)}"
        ) from None
    if not isinstance(target, str):
        raise TypeError(
            f"Argument to target must be a string, not {type(target)}"
        ) from None
    if target in _reference.SYSTEMS:
        if source == '1':
            return conversion_factory(source, '1')
        if quantity is None:
            if _is_ambiguous(source, target):
                raise UnitConversionError(
                    f"Conversion from {source!r} to {target!r}"
                    " is ambiguous without knowledge of physical quantity"
                ) from None
            q = get_quantity(symbolic.expression(source))
            return converter_factory(source, quantity=str(q)).to(target)
        return converter_factory(source, quantity=quantity).to(target)
    return conversion_factory(source, target)


def _is_ambiguous(unit: str, system: str):
    """True if the conversion from `unit` to `system` is ambiguous.

    Notes
    -----
    - This function is an internal helper for `~convert`.
    - This function will return `False` if either there is exactly one defined
      conversion from `unit` in `system`, or `unit` is not a canonical unit in
      `system`. The latter case does not guarantee a successful conversion; it
      simply allows this function to exit early if it has no chance of
      identifying an ambiguous conversion.
    """
    units = EQUIVALENT['units']
    others = {s for s in _reference.SYSTEMS if s != system}
    expr = symbolic.expression(unit)
    for s in others:
        targets = {v[system] for v in units.values() if v[s] == expr}
        if len(targets) > 1:
            return True
    return False


class Properties:
    """Canonical properties of a quantity within a metric system."""

    def __init__(
        self,
        system: str,
        unit: typing.Union[str, Unit],
    ) -> None:
        self._system = system.lower()
        self._unit = unit_factory(unit)
        self._dimension = None

    @property
    def unit(self):
        """The canonical unit of this quantity in this metric system."""
        return self._unit

    @property
    def dimension(self):
        """The dimension of this quantity in this metric system."""
        if self._dimension is None:
            self._dimension = self.unit.dimensions[self._system]
        return self._dimension

    def __eq__(self, __o) -> bool:
        """True if two instances have equal units and dimensions."""
        try:
            equal = [
                getattr(self, attr) == getattr(__o, attr)
                for attr in ('unit', 'dimension')
            ]
            return all(equal)
        except AttributeError:
            return False

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"{self.__class__.__qualname__}({self})"

    def __str__(self) -> str:
        """A simplified representation of this object."""
        properties = ', '.join(
            f"{p}={str(getattr(self, p, None))!r}"
            for p in ['unit', 'dimension']
        )
        return f"{properties} [{self._system!r}]"


class Quantity:
    """A single metric quantity.

    An instance of this class represents the properties of the named metric
    quantity in all defined metric systems.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._units = None
        self._dimensions = None

    def __getitem__(self, system: str):
        """Get this quantity's representation in the named metric system."""
        try:
            unit = self.units[system.lower()]
            return Properties(system, unit)
        except KeyError as err:
            raise QuantityError(
                f"No properties available for {self.name!r} in {system!r}"
            ) from err

    def convert(self, unit: str) -> Converter:
        """Create a conversion handler for `unit`.

        Parameters
        ----------
        unit : string
            The unit of this quantity from which to convert.

        Returns
        -------
        `~Converter`
            An instance of the unit-conversion handler. The returned object
            supports conversion from `unit` to any unit for which the conversion
            is defined, or to the equivalent unit in a given metric system.

        Notes
        -----
        - This method is defined within the scope of a particular metric
          quantity because unit conversions are only defined relative to their
          respective quantities, even though two quantities may have identical
          conversions (e.g., frequency and vorticity). The distinction between
          quantities is necessary when dealing with CGS electromagnetic units.
          For example, 'cm / s' is the canonical unit of both conductance and
          velocity in CGS; converting to its equivalent MKS unit therefore
          requires knowledge of the relevant quantity.
        """
        return converter_factory(unit, self.name)

    _attrs = ('dimensions', 'units')

    def __eq__(self, other) -> bool:
        """Called to test equality via self == other.
        
        Two instances of this class are either identical, in which case they are
        triviall equal, or they represent distinct quantities, in which they are
        not equal. In addition, an instance of this class will compare equal to
        its case-insensitive name.

        See Also
        --------
        `~metric.Quantity.__or__` : test equivalence between quantities.
        """
        if isinstance(other, str):
            return other.lower() == self.name
        return other is self

    def __or__(self, other) -> bool:
        """Called to test equivalence via self | other.
        
        Two metric quantities are equivalent if their dimensions are equal in a
        given metric system. This operation thus provides a way to compare
        unequal quantities to determine if they are have a physical
        correspondence. For example, energy and work are distinct quantities,
        but they have identical dimensions and are linked through the
        work-energy theorem.

        This operation is only defined between two instances of this class.
        """
        if isinstance(other, Quantity):
            for system in _reference.SYSTEMS:
                if self[system].dimension != other[system].dimension:
                    return False
            return True
        return NotImplemented

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"{self.__class__.__qualname__}({self})"

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return self.name

    @property
    def units(self):
        """The unit of this quantity in each metric system."""
        if self._units is None:
            self._units = UNITS[self.name]
        return self._units

    @property
    def dimensions(self):
        """The dimension of this quantity in each metric system."""
        if self._dimensions is None:
            self._dimensions = DIMENSIONS[self.name]
        return self._dimensions

    @property
    def name(self):
        """The name of this physical quantity."""
        return self._name


_quantities = {}
"""Internal collection of singleton `~Quantity` instances."""

def quantity_factory(arg: typing.Union[str, Quantity]) -> Quantity:
    """Factory function for metric quantities.

    Parameters
    ----------
    arg : string or `~Quantity`.
        The metric quantity to represent. If `arg` is already an instance of
        `~Quantity`, this function will immediately return it.

    Returns
    -------
    `~Quantity`
    """
    if isinstance(arg, Quantity):
        return arg
    name = str(arg).lower()
    if available := _quantities.get(name):
        return available
    q = Quantity(name)
    _quantities[name] = q
    return q


class SearchError(KeyError):
    """Error while searching for a requested metric."""
    pass


class System(collections.abc.Mapping):
    """Representations of physical quantities within a given metric system."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._units = None
        self._dimensions = None

    def __len__(self) -> int:
        """The number of quantities defined in this metric system."""
        return _reference.QUANTITIES.__len__()

    def __iter__(self) -> typing.Iterator:
        """Iterate over defined metric quantities."""
        return _reference.QUANTITIES.__iter__()

    def __getitem__(self, key: typing.Union[str, Quantity]):
        """Get the metric for the requested quantity in this system."""
        try:
            found = quantity_factory(key)
        except ValueError as err:
            raise KeyError(f"No known quantity called '{key}'") from err
        else:
            return found[self.name]

    def get_dimension(self, this: typing.Union[str, Unit]):
        """Compute the dimension of `unit` in this metric system."""
        unit = unit_factory(this)
        expression = symbolic.expression('1')
        systems = set()
        for term in unit:
            named = NamedUnit(term.base)
            allowed = named.systems['allowed']
            dimension = (
                named.dimensions[self.name] if len(allowed) > 1
                else named.dimensions[allowed[0]]
            )
            expression *= symbolic.expression(dimension) ** term.exponent
            systems.update(allowed)
        if self.name in systems:
            return dimension_factory(expression)
        raise ValueError(
            f"Can't define dimension of {unit!r} in {self.name!r}"
        ) from None

    def get_unit(
        self,
        *,
        unit: typing.Union[str, Unit]=None,
        dimension: typing.Union[str, Dimension]=None,
        quantity: typing.Union[str, Quantity]=None,
    ) -> typing.Optional[Unit]:
        """Get a canonical unit from a given unit, dimension, or quantity.

        This method will search for the unit in the current metric system based
        on `unit`, `dimension`, or `quantity`, in that order. All arguments
        default to `None`. If `unit` is not `None`, this method will attempt to
        return the equivalent canonical unit; if either `dimension` or
        `quantity` is not `None`, this method will attempt to return the unique
        corresponding unit.

        Parameters
        ----------
        unit : string or Unit
            A unit of measure in any system.

        dimension : string or Dimension
            A physical dimension.

        quantity : string or Quantity
            A physical quantity.

        Returns
        -------
        Unit
            The corresponding unit in the current metric system.

        """
        methods = {
            k: getattr(self, f'_unit_from_{k}')
            for k in ('unit', 'dimension', 'quantity')
        }
        targets = {
            'unit': unit,
            'dimension': dimension,
            'quantity': quantity,
        }
        return self._get_unit(methods, targets)

    _ST = typing.Union[str, Unit, Dimension, Quantity]

    def _get_unit(
        self,
        methods: typing.Dict[str, typing.Callable[[_ST], Unit]],
        targets: typing.Dict[str, _ST],
    ) -> Unit:
        """Search logic for `get_unit`."""
        nonnull = {k: v for k, v in targets.items() if v}
        cases = [(methods[k], v) for k, v in nonnull.items()]
        for (method, arg) in cases:
            if str(arg) == '1':
                return unit_factory(self['identity'].unit)
            if result := method(arg):
                return result
        args = self._format_targets(nonnull)
        errmsg = f"Could not determine unit in {self.name} from {args}"
        raise SearchError(errmsg)

    def _format_targets(self, targets: typing.Dict[str, _ST]):
        """Format `get_unit` targets for printing."""
        if not targets:
            return "nothing"
        args = [f'{k}={v!r}' for k, v in targets.items()]
        if 0 < len(args) < 3:
            return ' or '.join(args)
        return f"{', '.join(str(arg) for arg in args[:-1])}, or {args[-1]}"

    def _unit_from_unit(
        self,
        target: typing.Union[str, Unit],
    ) -> Unit:
        """Get the canonical unit corresponding to the given unit."""
        unit = unit_factory(target)
        return unit.normalize(self.name)

    def _unit_from_dimension(
        self,
        target: typing.Union[str, Dimension],
    ) -> Unit:
        """Get the canonical unit corresponding to the given dimension."""
        for quantity, dimension in self.dimensions.items():
            if dimension == target:
                return unit_factory(self.units[quantity])

    def _unit_from_quantity(
        self,
        quantity: typing.Union[str, Quantity],
    ) -> Unit:
        """Get the canonical unit corresponding to the given quantity."""
        return unit_factory(self[quantity].unit)

    def __eq__(self, other) -> bool:
        """True if two systems have the same `name` attribute."""
        if isinstance(other, System):
            return self.name == other.name.lower()
        if isinstance(other, str):
            return self.name == other.lower()
        return NotImplemented

    def __bool__(self) -> bool:
        """A defined metric system is always truthy."""
        return True

    def keys(self) -> typing.KeysView[str]:
        return super().keys()

    def values(self) -> typing.ValuesView[Properties]:
        return super().values()

    def items(self) -> typing.ItemsView[str, Properties]:
        return super().items()

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return str(self.name)

    @property
    def units(self):
        """The unit of each physical quantity in the metric system."""
        if self._units is None:
            self._units = CANONICAL['units'][self.name]
        return self._units

    @property
    def dimensions(self):
        """The dimension of each physical quantity in the metric system."""
        if self._dimensions is None:
            self._dimensions = CANONICAL['dimensions'][self.name]
        return self._dimensions

    @property
    def name(self):
        """The name of this metric system."""
        return self._name


_systems = {}
"""Internal collection of singleton `~System` instances."""

def system_factory(arg: typing.Union[str, System]) -> System:
    """Factory function for metric-system representations.

    Parameters
    ----------
    arg : string or `~System`.
        The metric system to represent. If `arg` is already an instance of
        `~System`, this function will immediately return it.

    Returns
    -------
    `~System`
    """
    if isinstance(arg, System):
        return arg
    name = str(arg).lower()
    if available := _systems.get(name):
        return available
    s = System(name)
    _systems[name] = s
    return s


