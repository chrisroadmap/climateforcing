import numbers

import numpy as np
import pytest

from climateforcing.utci import (
    mean_radiant_temperature,
    saturation_specific_humidity,
    utci,
)


def test_utci_array():
    EXPECTED_RESULT = np.array([19.62569676, 21.23458492])
    TEST_RESULT = utci(
        np.array([295, 296]),
        np.array([303, 304]),
        np.array([6.0, 6.0]),
        np.array([100, 100]),
    )
    assert np.allclose(TEST_RESULT, EXPECTED_RESULT)


def test_utci_list():
    EXPECTED_RESULT = np.array([19.62569676, 21.23458492])
    TEST_RESULT = utci([295, 296], [303, 304], [6.0, 6.0], [100, 100])
    assert np.allclose(TEST_RESULT, EXPECTED_RESULT)


def test_utci_scalar():
    EXPECTED_RESULT = 19.62569676
    TEST_RESULT = utci(295, 303, 6, 100)
    assert np.allclose(TEST_RESULT, EXPECTED_RESULT)
    assert isinstance(TEST_RESULT, numbers.Number)


def test_utci_raises():
    with pytest.raises(ValueError):
        utci(295, 296, 6, 100, humidity_type="wet")


def test_utci_specific_humidity():
    utci(295, 296, 6, 0.005, humidity_type="specific")


def test_ssh_array():
    EXPECTED_RESULT = np.array([1388.9644889401982, 3536.824257589194])
    TEST_RESULT = saturation_specific_humidity(np.array([285, 300]))
    assert np.allclose(TEST_RESULT, EXPECTED_RESULT)


def test_ssh_list():
    EXPECTED_RESULT = np.array([1388.9644889401982, 3536.824257589194])
    TEST_RESULT = saturation_specific_humidity([285, 300])
    assert np.allclose(TEST_RESULT, EXPECTED_RESULT)


def test_ssh_scalar():
    EXPECTED_RESULT = 1388.9644889401982
    TEST_RESULT = saturation_specific_humidity(285)
    assert np.allclose(TEST_RESULT, EXPECTED_RESULT)
    assert isinstance(TEST_RESULT, numbers.Number)


def test_mrt_array():
    EXPECTED_RESULT = np.array([313.33875095746555, 291.2519186818613])
    TEST_RESULT = mean_radiant_temperature(
        np.array([150, 50]),
        np.array([350, 150]),
        np.array([400, 200]),
        np.array([100, 50]),
        np.array([700, 400]),
        cos_zenith=np.array([0.5, 0.2]),
    )
    assert np.allclose(TEST_RESULT, EXPECTED_RESULT)


def test_mrt_list():
    EXPECTED_RESULT = np.array([313.33875095746555, 291.2519186818613])
    TEST_RESULT = mean_radiant_temperature(
        [150, 50], [350, 150], [400, 200], [100, 50], [700, 400], cos_zenith=[0.5, 0.2]
    )
    assert np.allclose(TEST_RESULT, EXPECTED_RESULT)


def test_mrt_scalar():
    EXPECTED_RESULT = 313.33875095746555
    TEST_RESULT = mean_radiant_temperature(150, 350, 400, 100, 700, cos_zenith=0.5)
    assert np.allclose(TEST_RESULT, EXPECTED_RESULT)
    assert isinstance(TEST_RESULT, numbers.Number)


def test_mrt_direct_exposed():
    # TODO: test output
    mean_radiant_temperature(
        150, 350, 400, 100, 700, cos_zenith=0.5, direct_exposed=0.7
    )


def test_integration_array():
    EXPECTED_RESULT = np.array([22.33807826, 18.04481664])
    mrt = mean_radiant_temperature(
        np.array([150, 50]),
        np.array([350, 150]),
        np.array([400, 200]),
        np.array([100, 50]),
        np.array([700, 400]),
        cos_zenith=np.array([0.5, 0.2]),
    )
    TEST_RESULT = utci(
        np.array([295, 296]), mrt, np.array([6.0, 6.0]), np.array([100, 100])
    )
    assert np.allclose(TEST_RESULT, EXPECTED_RESULT)


def test_integration_list():
    EXPECTED_RESULT = np.array([22.33807826, 18.04481664])
    mrt = mean_radiant_temperature(
        [150, 50], [350, 150], [400, 200], [100, 50], [700, 400], cos_zenith=[0.5, 0.2]
    )
    TEST_RESULT = utci([295, 296], mrt, [6.0, 6.0], [100, 100])
    assert np.allclose(TEST_RESULT, EXPECTED_RESULT)


def test_integration_scalar():
    EXPECTED_RESULT = 22.33807826100249
    mrt = mean_radiant_temperature(150, 350, 400, 100, 700, cos_zenith=0.5)
    TEST_RESULT = utci(295, mrt, 6, 100)
    assert np.allclose(TEST_RESULT, EXPECTED_RESULT)
    assert isinstance(TEST_RESULT, numbers.Number)
