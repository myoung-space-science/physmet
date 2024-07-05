import pytest

import support
from physmet import measurable
from physmet import measured
from physmet import numeric


def test_parse(measurables):
    """Test the function that attempts to parse a measurable object."""
    for case in measurables:
        result = measurable.parse(case['test'])
        expected = case['full']
        assert result == expected
    for case in measurables:
        result = measurable.parse(case['test'], distribute=True)
        expected = case['dist']
        assert result == expected
    assert measurable.parse(0) == (0, '1') # zero is measurable!
    with pytest.raises(measurable.ParsingTypeError):
        measurable.parse(None)
    with pytest.raises(measurable.ParsingTypeError):
        measurable.parse(slice(None))
    with pytest.raises(measurable.ParsingValueError):
        measurable.parse([1.1, 'm', 2.3, 'cm'])
    with pytest.raises(measurable.ParsingValueError):
        measurable.parse([(1.1, 'm'), (2.3, 5.8, 'cm')])
    with pytest.raises(measurable.ParsingTypeError):
        measurable.parse([1, (1, 'm')])


def test_measure(measurables):
    """Test the function that creates a measurement from measurable input."""
    for case in measurables:
        result = measurable.measure(case['test'])
        assert isinstance(result, numeric.Measurement)
        assert tuple(result.data) == case['full'][:-1]
        assert result.unit == case['full'][-1]
        assert measurable.measure(result) is result
    expected = measured.sequence([1], unit='m / s')
    this = support.Measurable([1], unit='m / s')
    assert measurable.measure(this) == expected
    expected = measured.sequence([1.1, 2.3], unit='m')
    strings = ['1.1', '2.3', 'm']
    assert measurable.measure(strings) == expected
    expected = measured.sequence([1.1, 2.3], unit='1')
    assert measurable.measure(strings[:-1]) == expected
    expected = measured.sequence([1.1], unit='m')
    assert measurable.measure('1.1', 'm') == expected
    expected = measured.sequence([1.1], unit='1')
    assert measurable.measure('1.1') == expected
    with pytest.raises(measurable.MeasuringTypeError):
        measurable.measure()
    with pytest.raises(measurable.MeasuringTypeError):
        measurable.measure(1, (1, 'm'))


