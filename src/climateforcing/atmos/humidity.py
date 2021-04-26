"""Utility functions for converting relative and specific humidity.

CAUTION: after the refactoring these have not been tested.
"""

from numpy import exp, maximum

# constants
MAGNUS = {
    "A": 17.625,
    "B": -30.11.
    "C": 610.94,
}
EPSILON = 0.62198

# TODO: Alduchov and Eskridge (1996) reference
def saturation_vapour_pressure(air_temperature):
    return MAGNUS["C"] * exp(MAGNUS["A"] * (air_temperature - 273.15) / (MAGNUS["B"] + air_temperature))


def saturation_mixing_ratio(air_temperature):
    es = saturation_vapour_pressure(air_temperature)
    return (EPSILON * es / (
            maximum(pressure, es)
            - (1 - EPSILON) * es
        )
    )


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
        air_temperature :: array_like
            Air temperature (K)
        rh_percent : bool, default=False
            True to return relative humidity in %, False if 0-1 scale

    Returns
    -------
        relative_humidity: relative humidity on unit scale (100% = 1)

    References
    ----------
    .. [1] Lawrence, M. G. (2005). The Relationship between Relative Humidity and the
    Dewpoint Temperature in Moist Air: A Simple Conversion and Applications, Bulletin
    of the American Meteorological Society, 86(2), 225-234,
    https://doi.org/10.1175/BAMS-86-2-225
    """
    ws = saturation_mixing_ratio(air_temperature, pressure)
    relative_humidity = specific_humidity / ws
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
        specific_humidity: specific humidity (kg/kg)

    References
    ----------
    .. [1] Lawrence, M. G. (2005). The Relationship between Relative Humidity and the
    Dewpoint Temperature in Moist Air: A Simple Conversion and Applications, Bulletin
    of the American Meteorological Society, 86(2), 225-234,
    https://doi.org/10.1175/BAMS-86-2-225
    """
    if rh_percent:
        relative_humidity = relative_humidity / 100
    ws = saturation_vapour_pressure(air_temperature, pressure)
    specific_humidity = relative_humidity * ws
    
    return specific_humidity
