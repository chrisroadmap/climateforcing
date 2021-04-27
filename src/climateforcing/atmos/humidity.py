"""Utility functions for converting relative and specific humidity.

CAUTION: after the refactoring these have not been tested.
"""

import numpy as np

EPSILON = 0.62198  # ratio of water vapour to dry air molecular weights

def calc_saturation_vapour_pressure(air_temperature):
    """Convert air temperature to saturation vapour pressure.

    Parameters
    ----------
        air_temperature : array_like
            air temperature, K

    Returns
    -------
        es : array_like
            saturation vapour pressure, Pa
    """
    # allow list input: convert to array
    # integers to negative powers not allowed, ensure float
    air_temperature = np.asarray(air_temperature).astype(float)

    log_es = (
        2.7150305 * np.log(air_temperature)
        + -2.8365744e3 * air_temperature ** (-2)
        + -6.028076559e3 * air_temperature ** (-1)
        + 1.954263612e1
        + -2.737830188e-2 * air_temperature
        + 1.6261698e-5 * air_temperature ** 2
        + 7.0229056e-10 * air_temperature ** 3
        + -1.8680009e-13 * air_temperature ** 4
    )

    return np.exp(log_es)


def calc_saturation_mixing_ratio(air_temperature, pressure):
    """Calculate saturation mixing ratio.

    Parameters
    ----------
        air_temperature : array_like
            Air temperature (K)
        pressure : array_like
            Air pressure (Pa)

    Returns
    -------
        saturation_mixing_ratio : array_like
            Saturation mixing ratio, dimensionless
    """
    saturation_vapour_pressure = calc_saturation_vapour_pressure(air_temperature)
    saturation_mixing_ratio = (
        EPSILON
        * saturation_vapour_pressure
        / (
            np.maximum(pressure, saturation_vapour_pressure)
            - (1 - EPSILON) * saturation_vapour_pressure
        )
    )
    return saturation_mixing_ratio

def specific_to_relative(
    specific_humidity,
    pressure=101325,
    air_temperature=288.15,
    rh_percent=False,
):
    """Convert specific humidity to relative humidity.

    Parameters
    ----------
        specific_humidity : array_like
            Specific humidity (kg/kg)
        pressure : array_like
            Air pressure (Pa)
        air_temperature : array_like
            Air temperature (K)
        rh_percent : bool, default=False
            True to return relative humidity in %, False if 0-1 scale

    Returns
    -------
        relative_humidity : array_like
            relative humidity
    """
    saturation_mixing_ratio = calc_saturation_mixing_ratio(air_temperature, pressure)
    relative_humidity = specific_humidity / saturation_mixing_ratio
    if rh_percent:
        relative_humidity = relative_humidity * 100
    return relative_humidity


def relative_to_specific(
    relative_humidity,
    pressure=101325,
    air_temperature=288.15,
    rh_percent=False,
):
    """Convert relative humidity to specific humidity.

    Parameters
    ----------
        relative_humidity : array_like
            Relative humidity, in either percent or 0-1 scale (see `rh_percent`)
        pressure : array_like
            Air pressure (Pa)
        air_temperature : array_like
            Air temperature (K)
        rh_percent : bool, default=False
            True if relative humidity is given in percent, False if 0-1 scale

    Returns
    -------
        specific_humidity: array_like
            specific humidity (kg/kg)
    """
    if rh_percent:
        relative_humidity = relative_humidity / 100
    saturation_mixing_ratio = calc_saturation_mixing_ratio(air_temperature, pressure)
    specific_humidity = relative_humidity * saturation_mixing_ratio
    return specific_humidity
