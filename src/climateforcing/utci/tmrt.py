"""
Calculate mean radiant temperature

Di Napoli, C., Hogan, R.J. & Pappenberger, F. Mean radiant temperature from 
global-scale numerical weather prediction models. Int J Biometeorol 64, 1233–1245
(2020). https://doi.org/10.1007/s00484-020-01900-5
"""

import numpy as np
from ..constants import STEFAN_BOLTZMANN


# TODO: can these inputs be arrays?
def mean_radiant_temperature(
    rlds, 
    rlus, 
    rsdsdiff, 
    rsus, 
    rsds,
    angle_factor_down=0.5,
    angle_factor_up=0.5,
    absorption=0.7,
    emissivity=0.97,
    direct_exposed=None,
    cos_zenith=1,
    lit=1
):
    """
    Inputs:
        rlds : float
            surface longwave downwelling radiation, W m-2
        rlus : float
            surface longwave upwelling radiation, W m-2
        rsdsdiff : float
            surface shortwave downwelling diffuse radiation, W m-2
        rsus : float
            surface shortwave upwelling radiation, W m-2
        rsds : float
            surface shortwave downwelling radiation, W m-2
        angle_factor_down : float
            proportion of the total view from the downwards direction
        angle_factor_up : float
            proportion of the total view from the upwards direction
        absorption: float
            absorption coefficient of the human body from shortwave radiation
        emissivity: float
            emissivity of the human body
        direct_exposed : float or None
            proportion of the body exposed to direct radiation. If None given,
            calculate it
        cos_zenith : float
            cosine of the solar zenith angle
        lit : float
            proportion of the time interval that the sun is above the horizon. Use
            lit=1 for instantaneous daytime calculations (this is most relevant for
            climate model data over longer periods like 3 hours).
    Returns:
        mean_radiant_temperature, Kelvin
    """
    # Calculate the direct normal radiation
    rsdsdirh = rsds - rsdsdiff
    #rsdsdirh[cosz==0] = 0
    if cos_zenith<=0:
        rsdsdirh = 0
        rsdsdir = 0
    else:
        rsdsdir = rsdsdirh/cos_zenith * lit
    # check here for monsters

    # calculate the direct exposed fraction if it is not given
    # no additional correction for lit fraction as it appears in rsdsdir
    if direct_exposed is None:
        zenith = np.degrees(np.arccos(cos_zenith))
        direct_exposed = 0.308 * np.cos(np.radians(90-zenith) * (0.998 - (90-zenith)**2/50000))

    return ((1/STEFAN_BOLTZMANN)*(angle_factor_down*rlds + angle_factor_up*rlus + absorption/emissivity*(angle_factor_down * rsdsdiff + angle_factor_up * rsus + direct_exposed*rsdsdir)))**(0.25)
