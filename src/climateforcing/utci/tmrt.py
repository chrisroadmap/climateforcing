"""
Calculate mean radiant temperature.

.. [1] Di Napoli, C., Hogan, R.J. & Pappenberger, F. Mean radiant temperature from
global-scale numerical weather prediction models. Int J Biometeorol 64, 1233–1245
(2020). https://doi.org/10.1007/s00484-020-01900-5
"""

import numpy as np

from ..constants import STEFAN_BOLTZMANN


# TODO: refactor, I'm too lazy/busy right now
def mean_radiant_temperature(  # pylint: disable=too-many-arguments,too-many-locals
    base,
    angle_factor_down=0.5,
    angle_factor_up=0.5,
    absorption=0.7,
    emissivity=0.97,
    direct_exposed=None,
    cos_zenith=1,
    lit=1,
):
    """Calculate the mean radiant temperature.

    Parameters
    ----------
        base : dict of array_like
            CMIP diagnostics required to calculate mean radiant temperature. The dict
            should contain 5 keys:

            rlds     : surface longwave downwelling radiation, W m-2
            rlus     : surface longwave upwelling radiation, W m-2
            rsds     : surface shortwave downwelling radiation, W m-2
            rsdsdiff : surface shortwave downwelling diffuse radiation, W m-2
            rsus     : surface shortwave upwelling radiation, W m-2

        angle_factor_down : float, default=0.5
            proportion of the total view from the downwards direction
        angle_factor_up : float, default=0.5
            proportion of the total view from the upwards direction
        absorption : float, default=0.7
            absorption coefficient of the human body from shortwave radiation
        emissivity : float, default=0.97
            emissivity of the human body
        direct_exposed : float or None
            proportion of the body exposed to direct radiation. If None given,
            calculate it
        cos_zenith : array_like, default=1
            cosine of the solar zenith angle
        lit : array_like, default=1
            proportion of the time interval that the sun is above the horizon. Use
            lit=1 for instantaneous daytime calculations (this is most relevant for
            climate model data over longer periods like 3 hours).

    Returns
    -------
        mean_radiant_temperature : array_like
            Mean radiant temperature in K
    """
    # this appears in APRP code twice: refactor target
    check_vars = ["rlds", "rlus", "rsds", "rsdsdiff", "rsus"]
    for check_var in check_vars:
        if check_var not in base.keys():
            raise ValueError("%s not present in %s" % (check_var, "base"))

    # check if the input is scalar or array
    base["rlds"] = np.asarray(base["rlds"])
    base["rlus"] = np.asarray(base["rlus"])
    base["rsdsdiff"] = np.asarray(base["rsdsdiff"])
    base["rsus"] = np.asarray(base["rsus"])
    base["rsds"] = np.asarray(base["rsds"])
    cos_zenith = np.asarray(cos_zenith)
    lit = np.asarray(lit)

    # refactor target
    for check_var in check_vars:
        if base[check_var].shape != base["rlds"].shape:
            raise ValueError(
                "%s %s in %s differs in shape to rlds %s"
                % (check_var, base[check_var].shape, "base", base["rlds"].shape,)
            )

    # > 0: one or more of the inputs are array so return array
    array_input = (
        base["rsds"].ndim
        + base["rlus"].ndim
        + base["rsdsdiff"].ndim
        + base["rsus"].ndim
        + base["rsds"].ndim
        + cos_zenith.ndim
        + lit.ndim
    )

    # Calculate the direct normal radiation
    base["rsdsdirh"] = base["rsds"] - base["rsdsdiff"]
    if base["rsdsdirh"].ndim == 0:
        base["rsdsdirh"] = base["rsdsdirh"][np.newaxis]
    if cos_zenith.ndim == 0:
        cos_zenith = cos_zenith[np.newaxis]
    if lit.ndim == 0:
        lit = lit * np.ones_like(cos_zenith)
    night = cos_zenith <= 0
    base["rsdsdirh"][night] = 0
    base["rsdsdir"] = np.zeros_like(cos_zenith)
    base["rsdsdir"][~night] = (
        base["rsdsdirh"][~night] / cos_zenith[~night] * lit[~night]
    )

    # calculate the direct exposed fraction if it is not given
    # no additional correction for lit fraction as it appears in rsdsdir
    if direct_exposed is None:
        zenith = np.degrees(np.arccos(cos_zenith))
        direct_exposed = 0.308 * np.cos(
            np.radians(90 - zenith) * (0.998 - (90 - zenith) ** 2 / 50000)
        )

    result = (
        (1 / STEFAN_BOLTZMANN)
        * (
            angle_factor_down * base["rlds"]
            + angle_factor_up * base["rlus"]
            + absorption
            / emissivity
            * (
                angle_factor_down * base["rsdsdiff"]
                + angle_factor_up * base["rsus"]
                + direct_exposed * base["rsdsdir"]
            )
        )
    ) ** (0.25)

    if not array_input:
        result = np.squeeze(result)[()]
    return result
