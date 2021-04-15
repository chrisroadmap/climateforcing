import pickle

import cftime
import numpy as np
import pytest

from climateforcing.solar.solar_position import (
    _apparent_sidereal_time,
    _apparent_sun_longitude,
    _check_and_expand_inputs,
    _day_of_year,
    _days_in_year,
    _earth_sun_distance,
    _geocentric_latitude,
    _geocentric_longitude,
    _geocentric_sun_declination,
    _modified_julian_day,
    _nutation,
    _observer_local_hour_angle,
    _sun_right_ascension,
    _topocentric_local_hour_angle,
    _topocentric_sun_coordinates,
    cos_mean_solar_zenith_angle,
    cos_solar_zenith_angle,
    modified_julian_date,
)

TEST_TIME = cftime.datetime(2021, 4, 6, 14, 52)
TEST_TIME_360 = cftime.datetime(2005, 2, 30, 12, calendar="360_day")
TEST_JDATE = 2459311.1194444443
LATITUDE_LEEDS = 53.8
LONGITUDE_LEEDS = -1.3
LATITUDE_GRID = np.arange(-90, 90.1, 5)
LONGITUDE_GRID = np.arange(0, 360, 10)


# Unit tests
def test_modified_julian_date():
    assert np.allclose(modified_julian_date(TEST_TIME), TEST_JDATE)


def test_apparent_sidereal_time():
    assert np.allclose(_apparent_sidereal_time(TEST_JDATE), 59.02818833717386)


def test_apparent_sun_longitude():
    assert np.allclose(_apparent_sun_longitude(TEST_JDATE), 17.020416164866194)


def test_check_and_expand_inputs():
    pass


def test_day_of_year():
    assert _day_of_year(TEST_TIME_360) == 59


def test_days_in_year():
    CALENDARS = [
        "standard",
        "gregorian",
        "proleptic_gregorian",
        "noleap",
        "365_day",
        "360_day",
        "julian",
        "all_leap",
        "366_day",
    ]
    EXPECTED_RESULTS = {
        "standard": [355, 366, 365, 366, 365, 366],
        "gregorian": [355, 366, 365, 366, 365, 366],
        "proleptic_gregorian": [365, 366, 365, 366, 365, 366],
        "noleap": [365, 365, 365, 365, 365, 365],
        "365_day": [365, 365, 365, 365, 365, 365],
        "360_day": [360, 360, 360, 360, 360, 360],
        "julian": [365, 366, 365, 366, 366, 366],
        "all_leap": [366, 366, 366, 366, 366, 366],
        "366_day": [366, 366, 366, 366, 366, 366],
    }
    for calendar in CALENDARS:
        for i, year in enumerate([1582, 2000, 2003, 2004, 2100, 2400]):
            assert (
                _days_in_year(year, calendar=calendar) == EXPECTED_RESULTS[calendar][i]
            )


def test_earth_sun_distance():
    assert np.allclose(_earth_sun_distance(TEST_JDATE), 1.0008310832150853)


def test_geocentric_latitude():
    assert np.allclose(_geocentric_latitude(TEST_JDATE), -5.290456650022293e-05)


def test_geocentric_longitude():
    assert np.allclose(_geocentric_longitude(TEST_JDATE), 17.031209617982313)


def test_geocentric_sun_declination():
    assert np.allclose(_geocentric_sun_declination(TEST_JDATE), 6.686302092071087)


def test_modified_julian_day():
    assert _modified_julian_day(TEST_TIME) == 2459311


def test_nutation():
    assert np.allclose(
        _nutation(TEST_JDATE), (-0.005106568279541951, 0.002725403033179875)
    )


def test_observer_local_hour_angle():
    assert np.allclose(
        _observer_local_hour_angle(TEST_JDATE, LONGITUDE_LEEDS), 42.040238394101415
    )


def test_sun_right_ascension():
    assert np.allclose(_sun_right_ascension(TEST_JDATE), 15.687949943072447)


def test_topocentric_local_hour_angle():
    assert np.allclose(
        _topocentric_local_hour_angle(TEST_JDATE, LATITUDE_LEEDS, LONGITUDE_LEEDS),
        42.09610782735281,
    )


def test_topocentric_sun_coordinates():
    assert np.allclose(
        _topocentric_sun_coordinates(TEST_JDATE, LATITUDE_LEEDS, LONGITUDE_LEEDS),
        (-0.05586943325139782, 6.574087560138782),
    )


def test_check_and_expand_inputs_raises():
    with pytest.raises(ValueError):
        _check_and_expand_inputs(LATITUDE_LEEDS, LONGITUDE_GRID)
    with pytest.raises(ValueError):
        _check_and_expand_inputs(
            LATITUDE_GRID.reshape((1, 37)), LONGITUDE_GRID.reshape((1, 36))
        )


def test_cos_mean_solar_zenith_angle():
    assert np.allclose(
        cos_mean_solar_zenith_angle(TEST_JDATE, 3, LATITUDE_LEEDS, LONGITUDE_LEEDS),
        (0.5166443273767068, 1),
    )
    with open("tests/testdata/solar/3hr_mean_cosz.pkl", "rb") as f:
        mean_cosz, lit = pickle.load(f)
    assert np.allclose(
        cos_mean_solar_zenith_angle(TEST_JDATE, 3, LATITUDE_GRID, LONGITUDE_GRID),
        (mean_cosz.T, lit.T),
    )


def test_cos_solar_zenith_angle():
    assert np.allclose(
        cos_solar_zenith_angle(TEST_JDATE, LATITUDE_LEEDS, LONGITUDE_LEEDS),
        0.5277476111683694,
    )
    with open("tests/testdata/solar/instant_cosz.pkl", "rb") as f:
        cosz = pickle.load(f)
    assert np.allclose(
        cos_solar_zenith_angle(TEST_JDATE, LATITUDE_GRID, LONGITUDE_GRID), cosz.T
    )
