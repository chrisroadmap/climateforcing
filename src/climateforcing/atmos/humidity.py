"""Utility functions for converting relative and specific humidity.

CAUTION: after the refactoring these have not been tested.
"""

from numpy import exp, ma, maximum

# TODO: tests
# TODO: why separate treatment for masked arrays?
# TODO: Alduchov and Eskridge (1996) reference


def specific_to_relative(
    prsaturation_vapour_pressuresure,
    specific_humidity,
    air_temperature,
    A=17.625,
    B=-30.11,
    C=610.94,
    masked=False,
):  # pylint: disable=invalid-name,too-many-arguments
    """Convert specific humidity to relative humidity.

    From Mark G. Lawrence, BAMS Feb 2005, eq. (6)

    Inputs:
        prsaturation_vapour_pressuresure :: float or `np.ndarray`
            Air prsaturation_vapour_pressuresure (Pa)
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
        saturation_vapour_pressure = C * exp(
            A * (air_temperature - 273.15) / (B + air_temperature)
        )
        saturation_mixing_ratio = (
            0.62198
            * saturation_vapour_pressure
            / (
                maximum(prsaturation_vapour_pressuresure, saturation_vapour_pressure)
                - (1 - 0.62198) * saturation_vapour_pressure
            )
        )
        relative_humidity = specific_humidity / saturation_mixing_ratio
    else:
        saturation_vapour_pressure = C * ma.exp(
            A * (air_temperature - 273.15) / (B + air_temperature)
        )
        saturation_mixing_ratio = (
            0.62198
            * saturation_vapour_pressure
            / (
                maximum(prsaturation_vapour_pressuresure, saturation_vapour_pressure)
                - (1 - 0.62198) * saturation_vapour_pressure
            )
        )
        relative_humidity = specific_humidity / saturation_mixing_ratio
    return relative_humidity


def relative_to_specific(
    prsaturation_vapour_pressuresure,
    relative_humidity,
    air_temperature,
    A=17.625,
    B=-30.11,
    C=610.94,
    masked=False,
):  # pylint: disable=invalid-name,too-many-arguments
    """Convert relative humidity to specific humidity.

    From Mark G. Lawrence, BAMS Feb 2005, eq. (6)

    Inputs:
        prsaturation_vapour_pressuresure :: float or `np.ndarray`
            Air prsaturation_vapour_pressuresure (Pa)
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
        saturation_vapour_pressure = C * exp(
            A * (air_temperature - 273.15) / (B + air_temperature)
        )
        specific_humidity = (
            0.62198
            * (relative_humidity * saturation_vapour_pressure)
            / (
                maximum(prsaturation_vapour_pressuresure, saturation_vapour_pressure)
                - (1 - 0.62198) * saturation_vapour_pressure
            )
        )
    else:
        saturation_vapour_pressure = C * ma.exp(
            A * (air_temperature - 273.15) / (B + air_temperature)
        )
        specific_humidity = (
            0.62198
            * (relative_humidity * saturation_vapour_pressure)
            / (
                maximum(prsaturation_vapour_pressuresure, saturation_vapour_pressure)
                - (1 - 0.62198) * saturation_vapour_pressure
            )
        )
    return specific_humidity
