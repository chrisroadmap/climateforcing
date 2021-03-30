"""Utility functions for converting relative and specific humidity.

CAUTION: after the refactoring these have not been tested.
"""

from numpy import exp, maximum

# TODO: Alduchov and Eskridge (1996) reference


def specific_to_relative(
    specific_humidity,
    pressure=101325,
    air_temperature=288.15,
    A=17.625,
    B=-30.11,
    C=610.94,
    rh_percent=False,
):  # pylint: disable=invalid-name,too-many-arguments
    """Convert specific humidity to relative humidity.

    From Mark G. Lawrence, BAMS Feb 2005, eq. (6)

    Inputs:
        specific_humidity :: float or `np.ndarray`
            Specific humidity (kg/kg)
        pressure :: float or `np.ndarray`
            Air prsaturation_vapour_pressuresure (Pa)
        air_temperature :: float or `np.ndarray`
            Air temperature (K)
        A, B, C :: float
            Fitting parameters from Alduchov and Eskridge (1996)
        rh_percent :: bool
            True to return relative humidity in %, False if 0-1 scale

    Returns:
        relative_humidity: relative humidity on unit scale (100% = 1)
    """
    saturation_vapour_pressure = C * exp(
        A * (air_temperature - 273.15) / (B + air_temperature)
    )
    saturation_mixing_ratio = (
        0.62198
        * saturation_vapour_pressure
        / (
            maximum(pressure, saturation_vapour_pressure)
            - (1 - 0.62198) * saturation_vapour_pressure
        )
    )
    relative_humidity = specific_humidity / saturation_mixing_ratio
    if rh_percent:
        relative_humidity = relative_humidity * 100
    return relative_humidity


def relative_to_specific(
    relative_humidity,
    pressure=101325,
    air_temperature=288.15,
    A=17.625,
    B=-30.11,
    C=610.94,
    rh_percent=False,
):  # pylint: disable=invalid-name,too-many-arguments
    """Convert relative humidity to specific humidity.

    From Mark G. Lawrence, BAMS Feb 2005, eq. (6)

    Inputs:
        relative_humidity :: float or `np.ndarray`
            relative humidity (see rh_percent)
        pressure :: float or `np.ndarray`
            Air prsaturation_vapour_pressuresure (Pa)
        air_temperature :: float or `np.ndarray`
            Air temperature (K)
        A, B, C :: float
            Fitting parameters from Alduchov and Eskridge (1996)
        rh_percent :: bool
            True if relative humidity is given in %, False if 0-1 scale

    Returns:
        specific_humidity: specific humidity (kg/kg)
    """
    if rh_percent:
        relative_humidity = relative_humidity / 100
    saturation_vapour_pressure = C * exp(
        A * (air_temperature - 273.15) / (B + air_temperature)
    )
    specific_humidity = (
        0.62198
        * (relative_humidity * saturation_vapour_pressure)
        / (
            maximum(pressure, saturation_vapour_pressure)
            - (1 - 0.62198) * saturation_vapour_pressure
        )
    )
    return specific_humidity
