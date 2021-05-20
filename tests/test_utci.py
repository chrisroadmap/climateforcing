import numpy as np
import pytest

from climateforcing.utci import (
    mean_radiant_temperature,
    universal_thermal_climate_index,
)


def test_utci_array():
    EXPECTED_RESULT = 273.15 + np.array([19.60850656, 21.2151128])
    TEST_RESULT = universal_thermal_climate_index(
        {
            "tas": np.array([295, 296]),
            "sfcWind": np.array([6.0, 6.0]),
            "hurs": np.array([100, 100]),
        },
        np.array([303, 304]),
    )
    assert np.allclose(TEST_RESULT, EXPECTED_RESULT)


def test_utci_huss():
    EXPECTED_RESULT = 273.15 + np.array([19.81876334, 20.98888025])
    TEST_RESULT = universal_thermal_climate_index(
        {
            "tas": np.array([295, 296]),
            "sfcWind": np.array([6.0, 6.0]),
            "huss": np.array([0.0167, 0.0167]),  # approx 100% RH at 295K, 1000hPa
        },
        np.array([303, 304]),
    )
    assert np.allclose(TEST_RESULT, EXPECTED_RESULT)


def test_utci_huss_ps():
    EXPECTED_RESULT = 273.15 + np.array([17.86596553, 17.84009998])
    TEST_RESULT = universal_thermal_climate_index(
        {
            "tas": np.array([295, 296]),
            "sfcWind": np.array([6.0, 6.0]),
            "huss": np.array([0.012, 0.010]),
            "ps": np.array([96000, 75000]),
        },
        np.array([303, 304]),
    )


def test_utci_raises():
    with pytest.raises(ValueError):
        universal_thermal_climate_index(
            {
                "tas": np.array([295, 296]),
                "sfcWind": np.array([6.0, 6.0]),
                "hurs": np.array([100, 100]),
                "huss": np.array([0.006, 0.006]),
            },
            np.array([303, 304]),
        )
    with pytest.raises(ValueError):
        universal_thermal_climate_index({}, np.array([303, 304]))


def test_mrt_array():
    EXPECTED_RESULT = np.array([313.33875095746555, 291.2519186818613])
    TEST_RESULT = mean_radiant_temperature(
        {
            "rlds": np.array([150, 50]),
            "rlus": np.array([350, 150]),
            "rsdsdiff": np.array([400, 200]),
            "rsus": np.array([100, 50]),
            "rsds": np.array([700, 400]),
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
                "rlds": 150,
                "rlus": np.array([350, 150]),
                "rsdsdiff": np.array([400, 200]),
                "rsus": np.array([100, 50]),
                "rsds": np.array([700, 400]),
            }
        )


def test_mrt_direct_exposed():
    # TODO: test output
    mean_radiant_temperature(
        {"rlds": 150, "rlus": 350, "rsdsdiff": 400, "rsus": 100, "rsds": 700},
        cos_zenith=0.5,
        direct_exposed=0.7,
    )


def test_integration_array():
    EXPECTED_RESULT = 273.15 + np.array([22.32159032, 18.02267449])
    mrt = mean_radiant_temperature(
        {
            "rlds": np.array([150, 50]),
            "rlus": np.array([350, 150]),
            "rsdsdiff": np.array([400, 200]),
            "rsus": np.array([100, 50]),
            "rsds": np.array([700, 400]),
        },
        cos_zenith=np.array([0.5, 0.2]),
        lit=np.array([1, 1]),
    )
    TEST_RESULT = universal_thermal_climate_index(
        {
            "tas": np.array([295, 296]),
            "sfcWind": np.array([6.0, 6.0]),
            "hurs": np.array([100, 100]),
        },
        mrt,
    )
    assert np.allclose(TEST_RESULT, EXPECTED_RESULT)
