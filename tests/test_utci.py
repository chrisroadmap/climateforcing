import numbers

import numpy as np
import pytest

from climateforcing.utci import (
    mean_radiant_temperature,
    saturation_specific_humidity,
    universal_thermal_climate_index,
)


def test_utci_array():
    EXPECTED_RESULT = 273.15 + np.array([19.62569676, 21.23458492])
    TEST_RESULT = universal_thermal_climate_index(
        {
            "tas" : np.array([295, 296]),
            "sfcWind" : np.array([6.0, 6.0]),
            "hurs" : np.array([100, 100]),
        },
        np.array([303, 304])
    )
    assert np.allclose(TEST_RESULT, EXPECTED_RESULT)


#def test_utci_list():
#    EXPECTED_RESULT = 273.15 + np.array([19.62569676, 21.23458492])
#    TEST_RESULT = universal_thermal_climate_index(
#        [295, 296], [303, 304], [6.0, 6.0], [100, 100]
#    )
#    assert np.allclose(TEST_RESULT, EXPECTED_RESULT)
#
#
#def test_utci_scalar():
#    EXPECTED_RESULT = 273.15 + 19.62569676
#    TEST_RESULT = universal_thermal_climate_index(295, 303, 6, 100)
#    assert np.allclose(TEST_RESULT, EXPECTED_RESULT)
#    assert isinstance(TEST_RESULT, numbers.Number)
#
#
def test_utci_raises():
    with pytest.raises(ValueError):
        universal_thermal_climate_index(
            {
                "tas" : np.array([295, 296]),
                "sfcWind" : np.array([6.0, 6.0]),
                "hurs" : np.array([100, 100]),
                "huss" : np.array([0.006, 0.006])
            },
            np.array([303, 304])
        )
    with pytest.raises(ValueError):
        universal_thermal_climate_index(
            {}, np.array([303, 304])
        )


def test_ssh_array():
    EXPECTED_RESULT = np.array([1388.9644889401982, 3536.824257589194])
    TEST_RESULT = saturation_specific_humidity(np.array([285, 300]))
    assert np.allclose(TEST_RESULT, EXPECTED_RESULT)


def test_mrt_array():
    EXPECTED_RESULT = np.array([313.33875095746555, 291.2519186818613])
    TEST_RESULT = mean_radiant_temperature(
        {
            "rlds" : np.array([150, 50]),
            "rlus" : np.array([350, 150]),
            "rsdsdiff" : np.array([400, 200]),
            "rsus" : np.array([100, 50]),
            "rsds" : np.array([700, 400]),
        },
        cos_zenith=np.array([0.5, 0.2]),
    )
    assert np.allclose(TEST_RESULT, EXPECTED_RESULT)


def test_mrt_raises():
    with pytest.raises(ValueError):
        mean_radiant_temperature({})
    with pytest.raises(ValueError):
        mean_radiant_temperature(
        {
            "rlds" : 150,
            "rlus" : np.array([350, 150]),
            "rsdsdiff" : np.array([400, 200]),
            "rsus" : np.array([100, 50]),
            "rsds" : np.array([700, 400]),
        }
    )


def test_mrt_direct_exposed():
    # TODO: test output
    mean_radiant_temperature(
        {
            "rlds" : 150,
            "rlus" : 350,
            "rsdsdiff" : 400,
            "rsus" : 100,
            "rsds" : 700,
        }, cos_zenith=0.5, direct_exposed=0.7
    )


def test_integration_array():
    EXPECTED_RESULT = 273.15 + np.array([22.33807826, 18.04481664])
    mrt = mean_radiant_temperature(
        {
            "rlds" : np.array([150, 50]),
            "rlus" : np.array([350, 150]),
            "rsdsdiff" : np.array([400, 200]),
            "rsus" : np.array([100, 50]),
            "rsds" : np.array([700, 400]),
        },
        cos_zenith=np.array([0.5, 0.2]),
        lit=np.array([1, 1]),
    )
    TEST_RESULT = universal_thermal_climate_index(
        {
            "tas" : np.array([295, 296]),
            "sfcWind": np.array([6.0, 6.0]),
            "hurs" : np.array([100, 100])
        },
        mrt
    )
    assert np.allclose(TEST_RESULT, EXPECTED_RESULT)
