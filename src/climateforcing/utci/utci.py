"""
Module to calculate Universal Thermal Climate Index.

Python translation from the original FORTRAN code by Peter Broede, used with
permission.

If you use this in your own work, please cite the following paper:

    Bröde P, Fiala D, Blazejczyk K, Holmér I, Jendritzky G, Kampmann B, Tinz B,
    Havenith G, 2012. Deriving the operational procedure for the Universal Thermal
    Climate Index (UTCI). International Journal of Biometeorology 56, 481-494,
    https://doi.org/10.1007/s00484-011-0454-1

The following is the original licence notice from the FORTRAN software.

---------------------------------------------------------------------------------------

    Version a 0.002, October 2009
    Changed ReadMe text and program messages for public release

    Copyright (C) 2009  Peter Broede

    The programs are distributed in the hope that they will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

Disclaimer of Warranty.

THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY APPLICABLE LAW. EXCEPT
WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES PROVIDE THE
PROGRAM "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING,
BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE. THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS
WITH YOU. SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY
SERVICING, REPAIR OR CORRECTION.

Limitation of Liability.

IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING WILL ANY COPYRIGHT
HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS THE PROGRAM AS PERMITTED ABOVE,
BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR
CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR INABILITY TO USE THE PROGRAM (INCLUDING
BUT NOT LIMITED TO LOSS OF DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY
YOU OR THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS),
EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

-----------------------------------------------
Peter Broede
Leibniz Research Centre for Working Environment and Human Factors (IfADo)
Leibniz-Institut für Arbeitsforschung an der TU Dortmund
Ardeystr. 67
D-44139 Dortmund
Fon: +49 +231 1084225
Fax: +49 +231 1084400
e-mail: broede@ifado.de
http://www.ifado.de
-----------------------------------------------
"""

import numpy as np

# TODO:
# - compare this to atmos.humidity code
# - throw warning if any of the input parameters are out of range the relationships
#   were designed for


def saturation_specific_humidity(air_temperature):
    """Convert air temperature to saturation specific humidity.

    Parameters
    ----------
        air_temperature : array_like
            air temperature, Kelvin

    Returns
    -------
        ssh array_like
            saturation specific humidity, Pa
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


def utci(
    air_temperature,
    mean_radiant_temperature,
    wind_speed_10m,
    humidity,
    humidity_type="relative",
):
    """Calculate Universal Thermal Climate Index.

    Parameters
    ----------
        air_temperature : array_like
            air temperature, K
        mean_radiant_temperature : array_like
            mean radiant temperature, K. See `utci.mean_radiant_temperature`.
        wind_speed_10m : array_like
            wind speed at 10m above ground level
        humidity : array_like
            either relative humidity in percent, or specific humidity in Pa. See
            `humidity_type`.
        humidity_type : {"relative", "specific"}

    Returns
    -------
        utci : array_like
            Universal Thermal Climate Index value in degrees Celcius scale

    Raises
    ------
        ValueError:
            if `humidity_type` does not begin with "r" or "s"
    """
    # allow list input: convert to array
    air_temperature = np.asarray(air_temperature)
    mean_radiant_temperature = np.asarray(mean_radiant_temperature)
    wind_speed_10m = np.asarray(wind_speed_10m)
    humidity = np.asarray(humidity)

    # turn off pylint warnings on short names, because the calculation gets stupid
    humidity_type = humidity_type.lower()[0]
    ta = air_temperature - 273.15  # pylint: disable=invalid-name
    es = saturation_specific_humidity(air_temperature)  # pylint: disable=invalid-name
    if humidity_type == "s":
        ws = 0.62198 * es / (es - (1 - 0.62198) * es)  # pylint: disable=invalid-name
        rh = 100 * humidity / ws  # pylint: disable=invalid-name
    elif humidity_type == "r":
        rh = humidity  # pylint: disable=invalid-name
    else:
        raise ValueError("`humidity_type` should be `relative` or `specific`")
    ppwv = es * rh / 100 / 1000  # partial pressure of water vapour, kPa
    va = wind_speed_10m  # pylint: disable=invalid-name
    delta_tmrt = mean_radiant_temperature - air_temperature

    # sixth order polynomial approximation to UTCI
    result = ta + (
        6.07562052e-01
        + (-2.27712343e-02) * ta
        + (8.06470249e-04) * ta * ta
        + (-1.54271372e-04) * ta * ta * ta
        + (-3.24651735e-06) * ta * ta * ta * ta
        + (7.32602852e-08) * ta * ta * ta * ta * ta
        + (1.35959073e-09) * ta * ta * ta * ta * ta * ta
        + (-2.25836520) * va
        + (8.80326035e-02) * ta * va
        + (2.16844454e-03) * ta * ta * va
        + (-1.53347087e-05) * ta * ta * ta * va
        + (-5.72983704e-07) * ta * ta * ta * ta * va
        + (-2.55090145e-09) * ta * ta * ta * ta * ta * va
        + (-7.51269505e-01) * va * va
        + (-4.08350271e-03) * ta * va * va
        + (-5.21670675e-05) * ta * ta * va * va
        + (1.94544667e-06) * ta * ta * ta * va * va
        + (1.14099531e-08) * ta * ta * ta * ta * va * va
        + (1.58137256e-01) * va * va * va
        + (-6.57263143e-05) * ta * va * va * va
        + (2.22697524e-07) * ta * ta * va * va * va
        + (-4.16117031e-08) * ta * ta * ta * va * va * va
        + (-1.27762753e-02) * va * va * va * va
        + (9.66891875e-06) * ta * va * va * va * va
        + (2.52785852e-09) * ta * ta * va * va * va * va
        + (4.56306672e-04) * va * va * va * va * va
        + (-1.74202546e-07) * ta * va * va * va * va * va
        + (-5.91491269e-06) * va * va * va * va * va * va
        + (3.98374029e-01) * delta_tmrt
        + (1.83945314e-04) * ta * delta_tmrt
        + (-1.73754510e-04) * ta * ta * delta_tmrt
        + (-7.60781159e-07) * ta * ta * ta * delta_tmrt
        + (3.77830287e-08) * ta * ta * ta * ta * delta_tmrt
        + (5.43079673e-10) * ta * ta * ta * ta * ta * delta_tmrt
        + (-2.00518269e-02) * va * delta_tmrt
        + (8.92859837e-04) * ta * va * delta_tmrt
        + (3.45433048e-06) * ta * ta * va * delta_tmrt
        + (-3.77925774e-07) * ta * ta * ta * va * delta_tmrt
        + (-1.69699377e-09) * ta * ta * ta * ta * va * delta_tmrt
        + (1.69992415e-04) * va * va * delta_tmrt
        + (-4.99204314e-05) * ta * va * va * delta_tmrt
        + (2.47417178e-07) * ta * ta * va * va * delta_tmrt
        + (1.07596466e-08) * ta * ta * ta * va * va * delta_tmrt
        + (8.49242932e-05) * va * va * va * delta_tmrt
        + (1.35191328e-06) * ta * va * va * va * delta_tmrt
        + (-6.21531254e-09) * ta * ta * va * va * va * delta_tmrt
        + (-4.99410301e-06) * va * va * va * va * delta_tmrt
        + (-1.89489258e-08) * ta * va * va * va * va * delta_tmrt
        + (8.15300114e-08) * va * va * va * va * va * delta_tmrt
        + (7.55043090e-04) * delta_tmrt * delta_tmrt
        + (-5.65095215e-05) * ta * delta_tmrt * delta_tmrt
        + (-4.52166564e-07) * ta * ta * delta_tmrt * delta_tmrt
        + (2.46688878e-08) * ta * ta * ta * delta_tmrt * delta_tmrt
        + (2.42674348e-10) * ta * ta * ta * ta * delta_tmrt * delta_tmrt
        + (1.54547250e-04) * va * delta_tmrt * delta_tmrt
        + (5.24110970e-06) * ta * va * delta_tmrt * delta_tmrt
        + (-8.75874982e-08) * ta * ta * va * delta_tmrt * delta_tmrt
        + (-1.50743064e-09) * ta * ta * ta * va * delta_tmrt * delta_tmrt
        + (-1.56236307e-05) * va * va * delta_tmrt * delta_tmrt
        + (-1.33895614e-07) * ta * va * va * delta_tmrt * delta_tmrt
        + (2.49709824e-09) * ta * ta * va * va * delta_tmrt * delta_tmrt
        + (6.51711721e-07) * va * va * va * delta_tmrt * delta_tmrt
        + (1.94960053e-09) * ta * va * va * va * delta_tmrt * delta_tmrt
        + (-1.00361113e-08) * va * va * va * va * delta_tmrt * delta_tmrt
        + (-1.21206673e-05) * delta_tmrt * delta_tmrt * delta_tmrt
        + (-2.18203660e-07) * ta * delta_tmrt * delta_tmrt * delta_tmrt
        + (7.51269482e-09) * ta * ta * delta_tmrt * delta_tmrt * delta_tmrt
        + (9.79063848e-11) * ta * ta * ta * delta_tmrt * delta_tmrt * delta_tmrt
        + (1.25006734e-06) * va * delta_tmrt * delta_tmrt * delta_tmrt
        + (-1.81584736e-09) * ta * va * delta_tmrt * delta_tmrt * delta_tmrt
        + (-3.52197671e-10) * ta * ta * va * delta_tmrt * delta_tmrt * delta_tmrt
        + (-3.36514630e-08) * va * va * delta_tmrt * delta_tmrt * delta_tmrt
        + (1.35908359e-10) * ta * va * va * delta_tmrt * delta_tmrt * delta_tmrt
        + (4.17032620e-10) * va * va * va * delta_tmrt * delta_tmrt * delta_tmrt
        + (-1.30369025e-09) * delta_tmrt * delta_tmrt * delta_tmrt * delta_tmrt
        + (4.13908461e-10) * ta * delta_tmrt * delta_tmrt * delta_tmrt * delta_tmrt
        + (9.22652254e-12) * ta * ta * delta_tmrt * delta_tmrt * delta_tmrt * delta_tmrt
        + (-5.08220384e-09) * va * delta_tmrt * delta_tmrt * delta_tmrt * delta_tmrt
        + (-2.24730961e-11)
        * ta
        * va
        * delta_tmrt
        * delta_tmrt
        * delta_tmrt
        * delta_tmrt
        + (1.17139133e-10) * va * va * delta_tmrt * delta_tmrt * delta_tmrt * delta_tmrt
        + (6.62154879e-10)
        * delta_tmrt
        * delta_tmrt
        * delta_tmrt
        * delta_tmrt
        * delta_tmrt
        + (4.03863260e-13)
        * ta
        * delta_tmrt
        * delta_tmrt
        * delta_tmrt
        * delta_tmrt
        * delta_tmrt
        + (1.95087203e-12)
        * va
        * delta_tmrt
        * delta_tmrt
        * delta_tmrt
        * delta_tmrt
        * delta_tmrt
        + (-4.73602469e-12)
        * delta_tmrt
        * delta_tmrt
        * delta_tmrt
        * delta_tmrt
        * delta_tmrt
        * delta_tmrt
        + (5.12733497) * ppwv
        + (-3.12788561e-01) * ta * ppwv
        + (-1.96701861e-02) * ta * ta * ppwv
        + (9.99690870e-04) * ta * ta * ta * ppwv
        + (9.51738512e-06) * ta * ta * ta * ta * ppwv
        + (-4.66426341e-07) * ta * ta * ta * ta * ta * ppwv
        + (5.48050612e-01) * va * ppwv
        + (-3.30552823e-03) * ta * va * ppwv
        + (-1.64119440e-03) * ta * ta * va * ppwv
        + (-5.16670694e-06) * ta * ta * ta * va * ppwv
        + (9.52692432e-07) * ta * ta * ta * ta * va * ppwv
        + (-4.29223622e-02) * va * va * ppwv
        + (5.00845667e-03) * ta * va * va * ppwv
        + (1.00601257e-06) * ta * ta * va * va * ppwv
        + (-1.81748644e-06) * ta * ta * ta * va * va * ppwv
        + (-1.25813502e-03) * va * va * va * ppwv
        + (-1.79330391e-04) * ta * va * va * va * ppwv
        + (2.34994441e-06) * ta * ta * va * va * va * ppwv
        + (1.29735808e-04) * va * va * va * va * ppwv
        + (1.29064870e-06) * ta * va * va * va * va * ppwv
        + (-2.28558686e-06) * va * va * va * va * va * ppwv
        + (-3.69476348e-02) * delta_tmrt * ppwv
        + (1.62325322e-03) * ta * delta_tmrt * ppwv
        + (-3.14279680e-05) * ta * ta * delta_tmrt * ppwv
        + (2.59835559e-06) * ta * ta * ta * delta_tmrt * ppwv
        + (-4.77136523e-08) * ta * ta * ta * ta * delta_tmrt * ppwv
        + (8.64203390e-03) * va * delta_tmrt * ppwv
        + (-6.87405181e-04) * ta * va * delta_tmrt * ppwv
        + (-9.13863872e-06) * ta * ta * va * delta_tmrt * ppwv
        + (5.15916806e-07) * ta * ta * ta * va * delta_tmrt * ppwv
        + (-3.59217476e-05) * va * va * delta_tmrt * ppwv
        + (3.28696511e-05) * ta * va * va * delta_tmrt * ppwv
        + (-7.10542454e-07) * ta * ta * va * va * delta_tmrt * ppwv
        + (-1.24382300e-05) * va * va * va * delta_tmrt * ppwv
        + (-7.38584400e-09) * ta * va * va * va * delta_tmrt * ppwv
        + (2.20609296e-07) * va * va * va * va * delta_tmrt * ppwv
        + (-7.32469180e-04) * delta_tmrt * delta_tmrt * ppwv
        + (-1.87381964e-05) * ta * delta_tmrt * delta_tmrt * ppwv
        + (4.80925239e-06) * ta * ta * delta_tmrt * delta_tmrt * ppwv
        + (-8.75492040e-08) * ta * ta * ta * delta_tmrt * delta_tmrt * ppwv
        + (2.77862930e-05) * va * delta_tmrt * delta_tmrt * ppwv
        + (-5.06004592e-06) * ta * va * delta_tmrt * delta_tmrt * ppwv
        + (1.14325367e-07) * ta * ta * va * delta_tmrt * delta_tmrt * ppwv
        + (2.53016723e-06) * va * va * delta_tmrt * delta_tmrt * ppwv
        + (-1.72857035e-08) * ta * va * va * delta_tmrt * delta_tmrt * ppwv
        + (-3.95079398e-08) * va * va * va * delta_tmrt * delta_tmrt * ppwv
        + (-3.59413173e-07) * delta_tmrt * delta_tmrt * delta_tmrt * ppwv
        + (7.04388046e-07) * ta * delta_tmrt * delta_tmrt * delta_tmrt * ppwv
        + (-1.89309167e-08) * ta * ta * delta_tmrt * delta_tmrt * delta_tmrt * ppwv
        + (-4.79768731e-07) * va * delta_tmrt * delta_tmrt * delta_tmrt * ppwv
        + (7.96079978e-09) * ta * va * delta_tmrt * delta_tmrt * delta_tmrt * ppwv
        + (1.62897058e-09) * va * va * delta_tmrt * delta_tmrt * delta_tmrt * ppwv
        + (3.94367674e-08) * delta_tmrt * delta_tmrt * delta_tmrt * delta_tmrt * ppwv
        + (-1.18566247e-09)
        * ta
        * delta_tmrt
        * delta_tmrt
        * delta_tmrt
        * delta_tmrt
        * ppwv
        + (3.34678041e-10)
        * va
        * delta_tmrt
        * delta_tmrt
        * delta_tmrt
        * delta_tmrt
        * ppwv
        + (-1.15606447e-10)
        * delta_tmrt
        * delta_tmrt
        * delta_tmrt
        * delta_tmrt
        * delta_tmrt
        * ppwv
        + (-2.80626406) * ppwv * ppwv
        + (5.48712484e-01) * ta * ppwv * ppwv
        + (-3.99428410e-03) * ta * ta * ppwv * ppwv
        + (-9.54009191e-04) * ta * ta * ta * ppwv * ppwv
        + (1.93090978e-05) * ta * ta * ta * ta * ppwv * ppwv
        + (-3.08806365e-01) * va * ppwv * ppwv
        + (1.16952364e-02) * ta * va * ppwv * ppwv
        + (4.95271903e-04) * ta * ta * va * ppwv * ppwv
        + (-1.90710882e-05) * ta * ta * ta * va * ppwv * ppwv
        + (2.10787756e-03) * va * va * ppwv * ppwv
        + (-6.98445738e-04) * ta * va * va * ppwv * ppwv
        + (2.30109073e-05) * ta * ta * va * va * ppwv * ppwv
        + (4.17856590e-04) * va * va * va * ppwv * ppwv
        + (-1.27043871e-05) * ta * va * va * va * ppwv * ppwv
        + (-3.04620472e-06) * va * va * va * va * ppwv * ppwv
        + (5.14507424e-02) * delta_tmrt * ppwv * ppwv
        + (-4.32510997e-03) * ta * delta_tmrt * ppwv * ppwv
        + (8.99281156e-05) * ta * ta * delta_tmrt * ppwv * ppwv
        + (-7.14663943e-07) * ta * ta * ta * delta_tmrt * ppwv * ppwv
        + (-2.66016305e-04) * va * delta_tmrt * ppwv * ppwv
        + (2.63789586e-04) * ta * va * delta_tmrt * ppwv * ppwv
        + (-7.01199003e-06) * ta * ta * va * delta_tmrt * ppwv * ppwv
        + (-1.06823306e-04) * va * va * delta_tmrt * ppwv * ppwv
        + (3.61341136e-06) * ta * va * va * delta_tmrt * ppwv * ppwv
        + (2.29748967e-07) * va * va * va * delta_tmrt * ppwv * ppwv
        + (3.04788893e-04) * delta_tmrt * delta_tmrt * ppwv * ppwv
        + (-6.42070836e-05) * ta * delta_tmrt * delta_tmrt * ppwv * ppwv
        + (1.16257971e-06) * ta * ta * delta_tmrt * delta_tmrt * ppwv * ppwv
        + (7.68023384e-06) * va * delta_tmrt * delta_tmrt * ppwv * ppwv
        + (-5.47446896e-07) * ta * va * delta_tmrt * delta_tmrt * ppwv * ppwv
        + (-3.59937910e-08) * va * va * delta_tmrt * delta_tmrt * ppwv * ppwv
        + (-4.36497725e-06) * delta_tmrt * delta_tmrt * delta_tmrt * ppwv * ppwv
        + (1.68737969e-07) * ta * delta_tmrt * delta_tmrt * delta_tmrt * ppwv * ppwv
        + (2.67489271e-08) * va * delta_tmrt * delta_tmrt * delta_tmrt * ppwv * ppwv
        + (3.23926897e-09)
        * delta_tmrt
        * delta_tmrt
        * delta_tmrt
        * delta_tmrt
        * ppwv
        * ppwv
        + (-3.53874123e-02) * ppwv * ppwv * ppwv
        + (-2.21201190e-01) * ta * ppwv * ppwv * ppwv
        + (1.55126038e-02) * ta * ta * ppwv * ppwv * ppwv
        + (-2.63917279e-04) * ta * ta * ta * ppwv * ppwv * ppwv
        + (4.53433455e-02) * va * ppwv * ppwv * ppwv
        + (-4.32943862e-03) * ta * va * ppwv * ppwv * ppwv
        + (1.45389826e-04) * ta * ta * va * ppwv * ppwv * ppwv
        + (2.17508610e-04) * va * va * ppwv * ppwv * ppwv
        + (-6.66724702e-05) * ta * va * va * ppwv * ppwv * ppwv
        + (3.33217140e-05) * va * va * va * ppwv * ppwv * ppwv
        + (-2.26921615e-03) * delta_tmrt * ppwv * ppwv * ppwv
        + (3.80261982e-04) * ta * delta_tmrt * ppwv * ppwv * ppwv
        + (-5.45314314e-09) * ta * ta * delta_tmrt * ppwv * ppwv * ppwv
        + (-7.96355448e-04) * va * delta_tmrt * ppwv * ppwv * ppwv
        + (2.53458034e-05) * ta * va * delta_tmrt * ppwv * ppwv * ppwv
        + (-6.31223658e-06) * va * va * delta_tmrt * ppwv * ppwv * ppwv
        + (3.02122035e-04) * delta_tmrt * delta_tmrt * ppwv * ppwv * ppwv
        + (-4.77403547e-06) * ta * delta_tmrt * delta_tmrt * ppwv * ppwv * ppwv
        + (1.73825715e-06) * va * delta_tmrt * delta_tmrt * ppwv * ppwv * ppwv
        + (-4.09087898e-07) * delta_tmrt * delta_tmrt * delta_tmrt * ppwv * ppwv * ppwv
        + (6.14155345e-01) * ppwv * ppwv * ppwv * ppwv
        + (-6.16755931e-02) * ta * ppwv * ppwv * ppwv * ppwv
        + (1.33374846e-03) * ta * ta * ppwv * ppwv * ppwv * ppwv
        + (3.55375387e-03) * va * ppwv * ppwv * ppwv * ppwv
        + (-5.13027851e-04) * ta * va * ppwv * ppwv * ppwv * ppwv
        + (1.02449757e-04) * va * va * ppwv * ppwv * ppwv * ppwv
        + (-1.48526421e-03) * delta_tmrt * ppwv * ppwv * ppwv * ppwv
        + (-4.11469183e-05) * ta * delta_tmrt * ppwv * ppwv * ppwv * ppwv
        + (-6.80434415e-06) * va * delta_tmrt * ppwv * ppwv * ppwv * ppwv
        + (-9.77675906e-06) * delta_tmrt * delta_tmrt * ppwv * ppwv * ppwv * ppwv
        + (8.82773108e-02) * ppwv * ppwv * ppwv * ppwv * ppwv
        + (-3.01859306e-03) * ta * ppwv * ppwv * ppwv * ppwv * ppwv
        + (1.04452989e-03) * va * ppwv * ppwv * ppwv * ppwv * ppwv
        + (2.47090539e-04) * delta_tmrt * ppwv * ppwv * ppwv * ppwv * ppwv
        + (1.48348065e-03) * ppwv * ppwv * ppwv * ppwv * ppwv * ppwv
    )
    return result
