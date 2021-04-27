import numpy as np

from climateforcing.atmos.humidity import relative_to_specific, specific_to_relative, calc_saturation_vapour_pressure

def test_saturation_vapour_pressure():
    EXPECTED_RESULT = np.array([1388.9644889401982, 3536.824257589194])
    TEST_RESULT = calc_saturation_vapour_pressure(np.array([285, 300]))
    assert np.allclose(TEST_RESULT, EXPECTED_RESULT)


def test_relative_to_specific():
    EXPECTED_RESULT = 0.006014444390991867
    test_result = relative_to_specific(0.5, air_temperature=290, pressure=1e5)
    assert test_result == EXPECTED_RESULT
    test_result = relative_to_specific(
        50, air_temperature=290, pressure=1e5, rh_percent=True
    )
    assert test_result == EXPECTED_RESULT


def test_specific_to_relative():
    EXPECTED_RESULT = 0.49879919157507707
    test_result = specific_to_relative(0.006, air_temperature=290, pressure=1e5)
    assert test_result == EXPECTED_RESULT
    test_result = specific_to_relative(
        0.006, air_temperature=290, pressure=1e5, rh_percent=True
    )
    assert test_result == EXPECTED_RESULT * 100
