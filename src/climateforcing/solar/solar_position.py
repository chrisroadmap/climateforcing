"""Module for calculating the instantanous and time-mean solar zenith angle."""

import cftime
import numpy as np

from .orbital_constants import (
    EARTH_SUN_DISTANCE_COEFFS,
    HELIOCENTRIC_LATITUDE_COEFFS,
    HELIOCENTRIC_LONGITUDE_COEFFS,
    NUTATION_COEFFICIENTS,
    NUTATION_SIN_TERMS,
)


def _days_in_year(year, calendar="standard"):
    return (
        cftime.datetime(year + 1, 1, 1, calendar=calendar).toordinal()
        - cftime.datetime(year, 1, 1, calendar=calendar).toordinal()
    )


def _day_of_year(cftime_in):
    return (
        cftime_in.toordinal()
        - cftime.datetime(cftime_in.year, 1, 1, calendar=cftime_in.calendar).toordinal()
    )


def _modified_julian_day(cftime_in):
    days_in_year_standard = _days_in_year(cftime_in.year)  # 365 or 366
    days_in_year_chosen = _days_in_year(
        cftime_in.year, cftime_in.calendar
    )  # how to handle 1582?
    doy = _day_of_year(cftime_in)
    year_offset = cftime.datetime(cftime_in.year, 1, 1).toordinal()
    year_fraction = doy * days_in_year_standard / days_in_year_chosen
    return int(np.round(year_offset + year_fraction))


def modified_julian_date(cftime_in):
    """Determine the Julian date to use for orbital parameters.

    If using the Gregorian, Proleptic Gregorian, standard, or Julian calendars, this
    will not actually modify anything. For 365, 366 and 360 day calendars used in some
    climate models, the Julian date is adjusted to try and get the solar position in
    the climate model to be roughly correct. It will introduce discrete jumps as the
    Julian day number may skip (in 360 and 365 day cases) or repeat (in 366 day cases)
    as we also want to ensure that the fractional part of the Julian date, representing
    the hour angle, is in the correct part of the diurnal cycle.

    Example
    -------
    >>> modified_julian_date(cftime.datetime(2021, 2, 30, 17, 36, calendar='360_day'))
    2459276.2333333334

    Parameters
    ----------
        cftime_in : :obj:`cftime.datetime`
            The date and time to calculate the orbital parameters for. See [1]_.

    Returns
    -------
        mjd : float
            modified julian date

    References
    ----------
    .. [1] cftime documentation, https://unidata.github.io/cftime/api.html
    """
    mjd = _modified_julian_day(cftime_in)
    adjustment = (
        cftime_in.toordinal(fractional=True)
        - cftime.datetime(
            cftime_in.year,
            cftime_in.month,
            cftime_in.day,
            12,
            0,
            0,
            calendar=cftime_in.calendar,
        ).toordinal()
    )
    return mjd + adjustment


def _term_sum(julian_date, coefficient_set):
    julian_millenium = (julian_date - 2451545) / 365250
    ncoeffs = len(coefficient_set)
    result = 0
    for icoeff in range(ncoeffs):
        coeff = np.sum(
            coefficient_set[icoeff][:, 0]
            * np.cos(
                coefficient_set[icoeff][:, 1]
                + coefficient_set[icoeff][:, 2] * julian_millenium
            )
        )
        result = result + 1e-8 * coeff * julian_millenium ** icoeff
    return result


def _geocentric_longitude(julian_date):
    result_radians = _term_sum(julian_date, HELIOCENTRIC_LONGITUDE_COEFFS)
    result_degrees = (180 + np.degrees(result_radians)) % 360
    return result_degrees


def _geocentric_latitude(julian_date):
    result_radians = _term_sum(julian_date, HELIOCENTRIC_LATITUDE_COEFFS)
    result_degrees = np.degrees(result_radians)
    return -result_degrees


def _earth_sun_distance(julian_date):
    # in AU
    return _term_sum(julian_date, EARTH_SUN_DISTANCE_COEFFS)


def _third_order(x, coeffs):  # pylint: disable=invalid-name
    return coeffs[0] + coeffs[1] * x + coeffs[2] * x * x + coeffs[3] * x * x * x


def _nutation(julian_date):
    julian_century = (julian_date - 2451545) / 36525
    nutation_x = np.array(
        [
            _third_order(
                julian_century, [297.85036, 445267.111480, -0.0019142, 1 / 189474]
            ),
            _third_order(
                julian_century, [357.52772, 35999.050340, -0.0001603, -1 / 300000]
            ),
            _third_order(
                julian_century, [134.96298, 477198.867398, -0.0086972, -1 / 56250]
            ),
            _third_order(
                julian_century, [93.27191, 483202.017538, -0.0036825, 1 / 327270]
            ),
            _third_order(
                julian_century, [125.04452, -1934.136261, 0.0020708, 1 / 450000]
            ),
        ]
    )
    nutation_longitude = (
        np.sum(
            (NUTATION_COEFFICIENTS[:, 0] + NUTATION_COEFFICIENTS[:, 1] * julian_century)
            * np.sin(np.sum(nutation_x * NUTATION_SIN_TERMS))
        )
        / 36000000
    )
    nutation_obliquity = (
        np.sum(
            (NUTATION_COEFFICIENTS[:, 2] + NUTATION_COEFFICIENTS[:, 3] * julian_century)
            * np.sin(np.sum(nutation_x * NUTATION_SIN_TERMS))
        )
        / 36000000
    )
    return nutation_longitude, nutation_obliquity


def _true_obliquity_of_ecliptic(julian_date):
    julian_10ka = (julian_date - 2451545) / 3652500
    mean_obliquity = (
        84381.448
        - 4680.93 * julian_10ka
        - 1.55 * julian_10ka ** 2
        + 1999.25 * julian_10ka ** 3
        - 51.38 * julian_10ka ** 4
        - 249.67 * julian_10ka ** 5
        - 39.05 * julian_10ka ** 6
        + 7.12 * julian_10ka ** 7
        + 27.87 * julian_10ka ** 8
        + 5.79 * julian_10ka ** 9
        + 2.45 * julian_10ka ** 10
    ) / 3600
    _, nutation_obliquity = _nutation(julian_date)
    return mean_obliquity + nutation_obliquity


def _apparent_sun_longitude(julian_date):
    delta_psi, _ = _nutation(julian_date)
    theta = _geocentric_longitude(julian_date)
    earth_sun = _earth_sun_distance(julian_date)
    delta_tau = -20.4898 / (3600 * earth_sun)
    return theta + delta_psi + delta_tau


def _apparent_sidereal_time(julian_date):
    julian_century = (julian_date - 2451545) / 36525
    mean_sidereal_time = (
        _third_order(
            julian_century,
            [280.46061837, 360.98564736629 * 36525, 0.000387933, 1 / 38710000],
        )
        % 360
    )
    delta_psi, _ = _nutation(julian_date)
    epsilon = _true_obliquity_of_ecliptic(julian_date)
    return mean_sidereal_time + delta_psi + np.cos(np.radians(epsilon))


def _sun_right_ascension(julian_date):
    lambd = np.radians(_apparent_sun_longitude(julian_date))
    epsilon = np.radians(_true_obliquity_of_ecliptic(julian_date))
    beta = np.radians(_geocentric_latitude(julian_date))
    return (
        np.degrees(
            np.arctan2(
                np.sin(lambd) * np.cos(epsilon) - np.tan(beta) * np.sin(epsilon),
                np.cos(lambd),
            )
        )
        % 360
    )


def _geocentric_sun_declination(julian_date):
    beta = np.radians(_geocentric_latitude(julian_date))
    epsilon = np.radians(_true_obliquity_of_ecliptic(julian_date))
    lambd = np.radians(_apparent_sun_longitude(julian_date))
    return np.degrees(
        np.arcsin(
            np.sin(beta) * np.cos(epsilon)
            + np.cos(beta) * np.sin(epsilon) * np.sin(lambd)
        )
    )


def _observer_local_hour_angle(julian_date, longitude):
    # longitude can be array or scalar
    ast = _apparent_sidereal_time(julian_date)
    alpha = _sun_right_ascension(julian_date)
    return ast + longitude - alpha


def _topocentric_sun_coordinates(julian_date, latitude, longitude):
    # Note here we do not adjust for elevation or refraction, because
    # we want TOA irradiance. The climate model will deal with the atmsopheric
    # radiative transfer.

    # We want function to make sense when fed with scalar, 1D lat and lon arrays,
    # or 2D lat-lon meshgrid.
    earth_sun = _earth_sun_distance(julian_date)
    equatorial_horizontal_parallax = 8.794 / (3600 * earth_sun)
    flatitude = np.arctan(0.99664719 * np.tan(np.radians(latitude)))
    flatx = np.cos(flatitude)
    flaty = 0.99664719 * np.sin(flatitude)
    hour_angle = _observer_local_hour_angle(julian_date, longitude)
    delta = _geocentric_sun_declination(julian_date)
    delta_alpha = np.degrees(
        np.arctan2(
            -flatx
            * np.sin(equatorial_horizontal_parallax)
            * np.sin(np.radians(hour_angle)),
            np.cos(np.radians(delta))
            - flatx
            * np.sin(equatorial_horizontal_parallax)
            * np.cos(np.radians(hour_angle)),
        )
    )
    delta_prime = np.degrees(
        np.arctan2(
            (np.sin(np.radians(delta)) - flaty * np.sin(equatorial_horizontal_parallax))
            * np.cos(delta_alpha),
            np.cos(np.radians(delta))
            - flaty
            * np.sin(equatorial_horizontal_parallax)
            * np.cos(np.radians(hour_angle)),
        )
    )
    return delta_alpha, delta_prime


def _topocentric_local_hour_angle(julian_date, latitude, longitude):
    hour_angle = _observer_local_hour_angle(julian_date, longitude)
    delta_alpha, _ = _topocentric_sun_coordinates(julian_date, latitude, longitude)
    return hour_angle - delta_alpha


def _check_and_expand_inputs(latitude, longitude):
    scalar_input = False
    latitude = np.asarray(latitude, dtype=float)
    longitude = np.asarray(longitude, dtype=float)
    if latitude.ndim == 2 and longitude.ndim == 2:  # assume meshgrid
        if latitude.shape != longitude.shape:
            raise ValueError(
                "For 2-dimensional input, latitude and longitude must be the same shape"
            )
    elif (latitude.ndim > 2 or longitude.ndim > 2) or (latitude.ndim != longitude.ndim):
        raise ValueError(
            "Latitude and longitude should both be scalars, 1-dimensional or "
            "2-dimensional arrays"
        )
    elif latitude.ndim == 1 and longitude.ndim == 1:
        latitude, longitude = np.meshgrid(latitude, longitude)
    elif latitude.ndim == 0 and longitude.ndim == 0:
        latitude = latitude[None]
        longitude = longitude[None]
        scalar_input = True
    return latitude, longitude, scalar_input


def cos_solar_zenith_angle(julian_date, latitude, longitude):
    """Calculate the cosine of the solar zenith angle.

    Parameters
    ----------
        julian_date : float
            julian date measured in days from 1 January 2000 12:00:00 GMT = 2451545.0.
            Designed to work with `modified_julian_date` for non-standard calendars.
        latitude : array_like
            latitude of the grid points to calculate solar zenith angle
        longitude: array_like
            longitude of the grid points to calculate solar zenith angle

    Returns
    -------
        cosz : array_like
            cosine of zenith angle.
    """
    latitude, longitude, scalar_input = _check_and_expand_inputs(latitude, longitude)
    _, delta_prime = _topocentric_sun_coordinates(julian_date, latitude, longitude)
    sin_declination = np.sin(np.radians(delta_prime))
    sin_latitude = np.sin(np.radians(latitude))
    cos_declination = np.cos(np.radians(delta_prime))
    cos_latitude = np.cos(np.radians(latitude))
    cos_hour = np.cos(
        np.radians(_topocentric_local_hour_angle(julian_date, latitude, longitude))
    )
    cosz = sin_declination * sin_latitude + cos_declination * cos_latitude * cos_hour
    if scalar_input:
        return cosz[0]
    return cosz


# this would be very difficult to refactor
def cos_mean_solar_zenith_angle(  # pylint: disable=too-many-locals,too-many-statements
    julian_date_mean, time_delta_hours, latitude, longitude
):
    """Calculate the cosine of the time-mean solar zenith angle.

    Parameters
    ------
        julian_date_mean : float
            julian date of the middle of the period, including fractional part of the
            day. Measured in days from 1 January 2000 12:00:00 GMT = 2451545.0
        time_delta_hours : float
            the length of time to calculate the time-mean solar zenith angle over.
            Results are generally stable for time_delta_hours <= 6. Therefore, the
            integration period is from julian_date_mean - time_delta_hours / 24 / 2
            to julian_date_mean + time_delta_hours / 24 / 2.
        latitude : array_like
            latitude of the grid points to calculate mean solar zenith angle
        longitude: array_like
            longitude of the grid points to calculate mean solar zenith angle

    Returns
    -------
        mean_cosz : array_like
            cosine of mean zenith angle.
        lit : array_like
            sunlit fraction of grid point over the time period.
    """
    # cases to consider:
    # 1. sun always above horizon: go from start to end of period
    #   1a. perpetual polar day
    #   1b. full daylight period outside polar regions
    # 2. sun sets, then rises in period (near polar day): two calculations, go from
    #    start of period to sunset, then sunrise to end of period, normalise by lit
    #    fraction
    # 3. sun rises: go from sunrise to end period and normalise by lit fraction
    # 4. sun sets: go from start of period to sunset and normalise by lit fraction
    # 5. sun rises then and sets in period (near polar night): go from sunrise to
    #    sunset, normalise by lit fraction
    # 6. sun always below horizon: zero
    #   6a. full nighttime period outside polar regions
    #   6b. perpetual polar night

    # 1/48 factor comes from 24 hours per day * half the distance from the middle of
    # the period to the start or end
    julian_date_start = julian_date_mean - time_delta_hours / 48
    julian_date_end = julian_date_mean + time_delta_hours / 48

    latitude, longitude, scalar_input = _check_and_expand_inputs(latitude, longitude)

    sin_latitude = np.sin(np.radians(latitude))
    cos_latitude = np.cos(np.radians(latitude))

    _, delta_prime = _topocentric_sun_coordinates(julian_date_mean, latitude, longitude)
    sin_declination = np.sin(np.radians(delta_prime))
    cos_declination = np.cos(np.radians(delta_prime))

    mean_cosz = np.zeros_like(latitude)
    lit = np.zeros_like(latitude)
    hour_lower = np.zeros_like(latitude)
    hour_upper = np.zeros_like(latitude)
    halfdaylength = np.zeros_like(latitude)
    hour_since_sunrise_start = np.zeros_like(latitude)
    hour_since_sunrise_end = np.zeros_like(latitude)
    hour_sunset = np.zeros_like(latitude)

    hour_start = np.radians(
        _topocentric_local_hour_angle(julian_date_start, latitude, longitude)
    )
    hour_end = np.radians(
        _topocentric_local_hour_angle(julian_date_end, latitude, longitude)
    )
    # sometimes because of periodicity hour_end < hour_start
    hour_end[hour_end < hour_start] = hour_end[hour_end < hour_start] + 2 * np.pi
    period = hour_end - hour_start
    cos_halfdaylength = (sin_latitude * sin_declination) / (
        cos_latitude * cos_declination
    )

    # case 6b
    polar_night = cos_halfdaylength < -1
    mean_cosz[polar_night] = 0
    lit[polar_night] = 0

    # case 1a
    polar_day = cos_halfdaylength > 1
    hour_lower[polar_day] = hour_start[polar_day]
    hour_upper[polar_day] = hour_start[polar_day] + period[polar_day]

    # For other cases, shift to counting times from sunrise to avoid overlap effects
    nonpolar = (-1 < cos_halfdaylength) & (cos_halfdaylength < 1)
    halfdaylength[nonpolar] = np.arccos(-cos_halfdaylength[nonpolar])
    hour_since_sunrise_start[nonpolar] = (
        hour_start[nonpolar] + halfdaylength[nonpolar]
    ) % (2 * np.pi)
    hour_since_sunrise_end[nonpolar] = (
        hour_since_sunrise_start[nonpolar] + period[nonpolar]
    ) % (2 * np.pi)
    hour_sunset[nonpolar] = 2 * halfdaylength[nonpolar]

    # cases 1b and 3
    case1b3 = (hour_since_sunrise_start <= hour_sunset) | (
        hour_since_sunrise_start < hour_since_sunrise_end
    )
    hour_lower[case1b3 & ~polar_day & ~polar_night] = (
        hour_since_sunrise_start[case1b3 & ~polar_day & ~polar_night]
        - halfdaylength[case1b3 & ~polar_day & ~polar_night]
    )
    hour_lower[(~case1b3) & ~polar_day & ~polar_night] = -halfdaylength[
        (~case1b3) & ~polar_day & ~polar_night
    ]

    # cases 1b and 4
    case1b4 = hour_since_sunrise_end <= hour_sunset
    hour_upper[case1b4 & ~polar_day & ~polar_night] = (
        hour_since_sunrise_end[case1b4 & ~polar_day & ~polar_night]
        - halfdaylength[case1b4 & ~polar_day & ~polar_night]
    )
    hour_upper[(~case1b4) & ~polar_day & ~polar_night] = (
        hour_sunset[(~case1b4) & ~polar_day & ~polar_night]
        - halfdaylength[(~case1b4) & ~polar_day & ~polar_night]
    )

    # case 6a
    case6a = (hour_since_sunrise_end > hour_since_sunrise_start) & (
        hour_since_sunrise_start > hour_sunset
    )
    mean_cosz[case6a] = 0
    lit[case6a] = 0

    delta_sin = np.sin(hour_upper) - np.sin(hour_lower)
    delta_hour = hour_upper - hour_lower

    # case 2
    case2 = (delta_hour < 0.0) & (~case6a)
    delta_sin[case2] = delta_sin[case2] + 2 * np.sqrt(
        1 - cos_halfdaylength[case2] ** 2
    )  # sin(halfdaylength)
    delta_hour[case2] = delta_hour[case2] + 2 * halfdaylength[case2]

    # cases 1b, 2, 3, 4, 5
    nonpolar_daytime = (~polar_night) & ~case6a
    mean_cosz[nonpolar_daytime] = (
        delta_sin[nonpolar_daytime]
        * cos_latitude[nonpolar_daytime]
        * cos_declination[nonpolar_daytime]
        / delta_hour[nonpolar_daytime]
        + sin_latitude[nonpolar_daytime] * sin_declination[nonpolar_daytime]
    )
    lit = delta_hour / period
    lit[lit < 0] = 0

    if scalar_input:
        return mean_cosz[0], lit[0]
    return mean_cosz, lit
