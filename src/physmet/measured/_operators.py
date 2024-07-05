from .. import data
from .. import numeric


def __eq__(a: numeric.MeasurementType, b) -> bool:
    """Called for a == b."""
    if isinstance(b, numeric.Measurement):
        return (
            data.isequal(a.data, b.data)
            and
            a.unit == b.unit
        )
    return False


def __repr__(a: numeric.MeasurementType) -> str:
    """Called for repr(a)."""
    return f"{a.__class__.__qualname__}({__str__(a)})"


def __str__(a: numeric.MeasurementType) -> str:
    """Called for str(a)."""
    return f"data={a.data}, unit={a.unit}"


