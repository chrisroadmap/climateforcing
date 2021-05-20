import numpy as np

from climateforcing.atmos.humidity import (
    calc_saturation_vapour_pressure,
    relative_to_specific,
    specific_to_relative,
)


def test_saturation_vapour_pressure():
    EXPECTED_RESULT = np.array([1386.30378655, 3527.7707872])
    TEST_RESULT = calc_saturation_vapour_pressure(np.array([285, 300]))
    assert np.allclose(TEST_RESULT, EXPECTED_RESULT)


def test_relative_to_specific():
    EXPECTED_RESULT = 0.0060003648770542905
    test_result = relative_to_specific(0.5, air_temperature=290, pressure=1e5)
    assert test_result == EXPECTED_RESULT
    test_result = relative_to_specific(
        50, air_temperature=290, pressure=1e5, rh_percent=True
    )
    assert test_result == EXPECTED_RESULT


def test_specific_to_relative():
    EXPECTED_RESULT = 0.49996959542779756
    test_result = specific_to_relative(0.006, air_temperature=290, pressure=1e5)
    assert test_result == EXPECTED_RESULT
    test_result = specific_to_relative(
        0.006, air_temperature=290, pressure=1e5, rh_percent=True
    )
    assert test_result == EXPECTED_RESULT * 100
