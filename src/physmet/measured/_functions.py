from .. import metric
from .. import quantity
from .. import numeric


def conversion(a: numeric.Measurement, unit):
    """Compute the numeric scale factor for a unit conversion, if possible."""
    if quantity.isunitlike(unit):
        try:
            return metric.conversion(a.unit, unit)
        except metric.UnitConversionError as err:
            if not (a.unit | unit):
                raise ValueError(
                    f"The unit {str(unit)!r} is inconsistent"
                    f" with {str(a.unit)!r}"
                ) from err
            raise err
    raise TypeError(
        f"Cannot interpret {unit!r} as a unit"
    ) from None


