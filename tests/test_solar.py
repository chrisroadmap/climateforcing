import cftime
import numpy as np

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
    _term_sum,
    _third_order,
    _topocentric_local_hour_angle,
    _topocentric_sun_coordinates,
    _true_obliquity_of_ecliptic,
    cos_mean_solar_zenith_angle,
    cos_solar_zenith_angle,
    modified_julian_date,
)

TEST_TIME = cftime.datetime(2021, 4, 6, 14, 52)
TEST_TIME_360 = cftime.datetime(2005, 2, 30, 12, calendar="360_day")

# Unit tests
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


def test_day_of_year():
    assert _day_of_year(TEST_TIME_360) == 59


def test_modified_julian_day():
    assert _modified_julian_day(TEST_TIME) == 2459311


def test_modified_julian_date():
    assert np.allclose(modified_julian_date(TEST_TIME), 2459311.1194444443)
