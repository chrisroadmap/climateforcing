"""
Utility functions for converting relative and specific humidity

CAUTION: after the refactoring these have not been tested - writing tests is a TODO
"""

from numpy import exp, ma, maximum

# TODO: tests
# TODO: why separate treatment for masked arrays?
# TODO: Alduchov and Eskridge (1996) reference


def specific_to_relative(pressure, specific_humidity, air_temperature, A=17.625, B=-30.11, C=610.94, masked=False):  # pylint: disable=invalid-name
    """
    From Mark G. Lawrence, BAMS Feb 2005, eq. (6)

    Inputs:
        pressure :: float or `np.ndarray` 
            Air pressure (Pa)
        specific_humidity :: float or `np.ndarray`
            Specific humidity (kg/kg)
        air_temperature :: float or `np.ndarray`
            Air temperature (K)
        A, B, C :: float
            Fitting parameters from Alduchov and Eskridge (1996)
        masked :: bool
            True if inputs are masked arrays (default False)

    Returns:
        relative_humidity: relative humidity on unit scale (100% = 1)
    """
    if not masked:
        es = C * exp(A * (air_temperature - 273.15) / (B + air_temperature))
        ws = 0.62198 * es / (maximum(pressure, es) - (1 - 0.62198) * es)
        relative_humidity = specific_humidity / ws
    else:
        es = C * ma.exp(A * (air_temperature - 273.15) / (B + air_temperature))
        ws = 0.62198 * es / (maximum(pressure, es) - (1 - 0.62198) * es)
        relative_humidity = specific_humidity / ws
    return relative_humidity


def relative_to_specific(pressure, relative_humidity, air_temperature, A=17.625, B=-30.11, C=610.94, masked=False):  # pylint: disable=invalid-name
    """
    From Mark G. Lawrence, BAMS Feb 2005, eq. (6)

    Inputs:
        pressure :: float or `np.ndarray`
            Air pressure (Pa)
        relative_humidity :: float or `np.ndarray`
            Specific humidity (kg/kg)
        air_temperature :: float or `np.ndarray`
            Air temperature (K)
        A, B, C :: float
            Fitting parameters from Alduchov and Eskridge (1996)
        masked :: bool
            True if inputs are masked arrays (default False)

    Returns:
        specific_humidity: relative humidity on unit scale (100% = 1)
    """
    if not masked:
        es = C * exp(A * (air_temperature - 273.15) / (B + air_temperature))
        q = 0.62198 * (relative_humidity * es) / (maximum(pressure, es) - (1 - 0.62198) * es)
    else:
        es = C * ma.exp(A * (air_temperature - 273.15) / (B + air_temperature))
        q = 0.62198 * (relative_humidity * es) / (maximum(pressure, es) - (1 - 0.62198) * es)
    return specific_humidity
