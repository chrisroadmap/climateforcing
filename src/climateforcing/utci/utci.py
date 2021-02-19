"""
Module to calculate Universal Thermal Climate Index

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
    """Conversion of air temperature to saturation specific humidity.

    Inputs:
        air_temperature :: float or `numpy.ndarray`
            air temperature, Kelvin

    Returns:
        ssh :: float or `numpy.ndarray`
            saturation specific humidity, Pa
    """
    # integers to negative powers not allowed, ensure float
    if isinstance(air_temperature, np.ndarray):
        air_temperature = air_temperature.astype(float)

    log_es = (
        2.7150305 * np.log(air_temperature) + 
        -2.8365744E3 * air_temperature**(-2) + 
        -6.028076559E3 * air_temperature**(-1) + 
        1.954263612E1 + 
        -2.737830188E-2 * air_temperature + 
        1.6261698E-5 * air_temperature**2 + 
        7.0229056E-10 * air_temperature**3 + 
        -1.8680009E-13 * air_temperature**4
    )
    
    return np.exp(log_es)


def utci(air_temperature, mean_radiant_temperature, wind_speed_10m, humidity,
    humidity_type='relative'):
    """Calculate Universal Thermal Climate Index

    Inputs:
        air_temperature :: float or `numpy.ndarray`
            air temperature, Kelvin
        mean_radiant_temperature :: float or `numpy.ndarray`
            mean radiant temperature, Kelvin. See `utci.mean_radiant_temperature`.
        wind_speed_10m :: float or `numpy.ndarray`
            wind speed at 10m above ground level
        humidity :: float or `numpy.ndarray`
            either relative humidity in percent, or specific humidity in Pa. See
            `humidity_type`.
        humidity_type :: string
            'relative' : humidity is expressed as relative humidity
            'specific' : humidity is expressed as specific humidity

    Returns:
        utci :: float
            Universal Thermal Climate Index value in degrees Celcius scale

    Raises:
        ValueError if `humidity_type` does not begin with 'r' or 's'
    """

    ht = humidity_type.lower()[0]
    ta = air_temperature - 273.15
    es = saturation_specific_humidity(air_temperature)
    if ht=='s':
        ws = 0.62198 * es/(es - (1 - 0.62198) * es)
        rh = 100 * humidity/ws
    elif ht=='r':
        rh = humidity
    else:
        raise ValueError('`humidity_type` should be `relative` or `specific`')
    ppwv = es * rh/100 / 1000   # partial pressure of water vapour, kPa
    va = wind_speed_10m
    delta_tmrt = mean_radiant_temperature - air_temperature

    # sixth order polynomial approximation to UTCI
    result = ta + (
        6.07562052E-01 +
        ( -2.27712343E-02 ) * ta + 
        ( 8.06470249E-04 ) * ta*ta + 
        ( -1.54271372E-04 ) * ta*ta*ta + 
        ( -3.24651735E-06 ) * ta*ta*ta*ta + 
        ( 7.32602852E-08 ) * ta*ta*ta*ta*ta + 
        ( 1.35959073E-09 ) * ta*ta*ta*ta*ta*ta + 
        ( -2.25836520 ) * va + 
        ( 8.80326035E-02 ) * ta*va + 
        ( 2.16844454E-03 ) * ta*ta*va + 
        ( -1.53347087E-05 ) * ta*ta*ta*va + 
        ( -5.72983704E-07 ) * ta*ta*ta*ta*va + 
        ( -2.55090145E-09 ) * ta*ta*ta*ta*ta*va + 
        ( -7.51269505E-01 ) * va*va + 
        ( -4.08350271E-03 ) * ta*va*va + 
        ( -5.21670675E-05 ) * ta*ta*va*va + 
        ( 1.94544667E-06 ) * ta*ta*ta*va*va + 
        ( 1.14099531E-08 ) * ta*ta*ta*ta*va*va + 
        ( 1.58137256E-01 ) * va*va*va + 
        ( -6.57263143E-05 ) * ta*va*va*va + 
        ( 2.22697524E-07 ) * ta*ta*va*va*va + 
        ( -4.16117031E-08 ) * ta*ta*ta*va*va*va + 
        ( -1.27762753E-02 ) * va*va*va*va + 
        ( 9.66891875E-06 ) * ta*va*va*va*va + 
        ( 2.52785852E-09 ) * ta*ta*va*va*va*va + 
        ( 4.56306672E-04 ) * va*va*va*va*va + 
        ( -1.74202546E-07 ) * ta*va*va*va*va*va + 
        ( -5.91491269E-06 ) * va*va*va*va*va*va + 
        ( 3.98374029E-01 ) * delta_tmrt + 
        ( 1.83945314E-04 ) * ta*delta_tmrt + 
        ( -1.73754510E-04 ) * ta*ta*delta_tmrt + 
        ( -7.60781159E-07 ) * ta*ta*ta*delta_tmrt + 
        ( 3.77830287E-08 ) * ta*ta*ta*ta*delta_tmrt + 
        ( 5.43079673E-10 ) * ta*ta*ta*ta*ta*delta_tmrt + 
        ( -2.00518269E-02 ) * va*delta_tmrt + 
        ( 8.92859837E-04 ) * ta*va*delta_tmrt + 
        ( 3.45433048E-06 ) * ta*ta*va*delta_tmrt + 
        ( -3.77925774E-07 ) * ta*ta*ta*va*delta_tmrt + 
        ( -1.69699377E-09 ) * ta*ta*ta*ta*va*delta_tmrt + 
        ( 1.69992415E-04 ) * va*va*delta_tmrt + 
        ( -4.99204314E-05 ) * ta*va*va*delta_tmrt + 
        ( 2.47417178E-07 ) * ta*ta*va*va*delta_tmrt + 
        ( 1.07596466E-08 ) * ta*ta*ta*va*va*delta_tmrt + 
        ( 8.49242932E-05 ) * va*va*va*delta_tmrt + 
        ( 1.35191328E-06 ) * ta*va*va*va*delta_tmrt + 
        ( -6.21531254E-09 ) * ta*ta*va*va*va*delta_tmrt + 
        ( -4.99410301E-06 ) * va*va*va*va*delta_tmrt + 
        ( -1.89489258E-08 ) * ta*va*va*va*va*delta_tmrt + 
        ( 8.15300114E-08 ) * va*va*va*va*va*delta_tmrt + 
        ( 7.55043090E-04 ) * delta_tmrt*delta_tmrt + 
        ( -5.65095215E-05 ) * ta*delta_tmrt*delta_tmrt + 
        ( -4.52166564E-07 ) * ta*ta*delta_tmrt*delta_tmrt + 
        ( 2.46688878E-08 ) * ta*ta*ta*delta_tmrt*delta_tmrt + 
        ( 2.42674348E-10 ) * ta*ta*ta*ta*delta_tmrt*delta_tmrt + 
        ( 1.54547250E-04 ) * va*delta_tmrt*delta_tmrt + 
        ( 5.24110970E-06 ) * ta*va*delta_tmrt*delta_tmrt + 
        ( -8.75874982E-08 ) * ta*ta*va*delta_tmrt*delta_tmrt + 
        ( -1.50743064E-09 ) * ta*ta*ta*va*delta_tmrt*delta_tmrt + 
        ( -1.56236307E-05 ) * va*va*delta_tmrt*delta_tmrt + 
        ( -1.33895614E-07 ) * ta*va*va*delta_tmrt*delta_tmrt + 
        ( 2.49709824E-09 ) * ta*ta*va*va*delta_tmrt*delta_tmrt + 
        ( 6.51711721E-07 ) * va*va*va*delta_tmrt*delta_tmrt + 
        ( 1.94960053E-09 ) * ta*va*va*va*delta_tmrt*delta_tmrt + 
        ( -1.00361113E-08 ) * va*va*va*va*delta_tmrt*delta_tmrt + 
        ( -1.21206673E-05 ) * delta_tmrt*delta_tmrt*delta_tmrt + 
        ( -2.18203660E-07 ) * ta*delta_tmrt*delta_tmrt*delta_tmrt + 
        ( 7.51269482E-09 ) * ta*ta*delta_tmrt*delta_tmrt*delta_tmrt + 
        ( 9.79063848E-11 ) * ta*ta*ta*delta_tmrt*delta_tmrt*delta_tmrt + 
        ( 1.25006734E-06 ) * va*delta_tmrt*delta_tmrt*delta_tmrt + 
        ( -1.81584736E-09 ) * ta*va*delta_tmrt*delta_tmrt*delta_tmrt + 
        ( -3.52197671E-10 ) * ta*ta*va*delta_tmrt*delta_tmrt*delta_tmrt + 
        ( -3.36514630E-08 ) * va*va*delta_tmrt*delta_tmrt*delta_tmrt + 
        ( 1.35908359E-10 ) * ta*va*va*delta_tmrt*delta_tmrt*delta_tmrt + 
        ( 4.17032620E-10 ) * va*va*va*delta_tmrt*delta_tmrt*delta_tmrt + 
        ( -1.30369025E-09 ) * delta_tmrt*delta_tmrt*delta_tmrt*delta_tmrt + 
        ( 4.13908461E-10 ) * ta*delta_tmrt*delta_tmrt*delta_tmrt*delta_tmrt + 
        ( 9.22652254E-12 ) * ta*ta*delta_tmrt*delta_tmrt*delta_tmrt*delta_tmrt + 
        ( -5.08220384E-09 ) * va*delta_tmrt*delta_tmrt*delta_tmrt*delta_tmrt + 
        ( -2.24730961E-11 ) * ta*va*delta_tmrt*delta_tmrt*delta_tmrt*delta_tmrt + 
        ( 1.17139133E-10 ) * va*va*delta_tmrt*delta_tmrt*delta_tmrt*delta_tmrt + 
        ( 6.62154879E-10 ) * delta_tmrt*delta_tmrt*delta_tmrt*delta_tmrt*delta_tmrt + 
        ( 4.03863260E-13 ) * ta*delta_tmrt*delta_tmrt*delta_tmrt*delta_tmrt*delta_tmrt + 
        ( 1.95087203E-12 ) * va*delta_tmrt*delta_tmrt*delta_tmrt*delta_tmrt*delta_tmrt + 
        ( -4.73602469E-12 ) * delta_tmrt*delta_tmrt*delta_tmrt*delta_tmrt*delta_tmrt*delta_tmrt + 
        ( 5.12733497 ) * ppwv + 
        ( -3.12788561E-01 ) * ta*ppwv + 
        ( -1.96701861E-02 ) * ta*ta*ppwv + 
        ( 9.99690870E-04 ) * ta*ta*ta*ppwv + 
        ( 9.51738512E-06 ) * ta*ta*ta*ta*ppwv + 
        ( -4.66426341E-07 ) * ta*ta*ta*ta*ta*ppwv + 
        ( 5.48050612E-01 ) * va*ppwv + 
        ( -3.30552823E-03 ) * ta*va*ppwv + 
        ( -1.64119440E-03 ) * ta*ta*va*ppwv + 
        ( -5.16670694E-06 ) * ta*ta*ta*va*ppwv + 
        ( 9.52692432E-07 ) * ta*ta*ta*ta*va*ppwv + 
        ( -4.29223622E-02 ) * va*va*ppwv + 
        ( 5.00845667E-03 ) * ta*va*va*ppwv + 
        ( 1.00601257E-06 ) * ta*ta*va*va*ppwv + 
        ( -1.81748644E-06 ) * ta*ta*ta*va*va*ppwv + 
        ( -1.25813502E-03 ) * va*va*va*ppwv + 
        ( -1.79330391E-04 ) * ta*va*va*va*ppwv + 
        ( 2.34994441E-06 ) * ta*ta*va*va*va*ppwv + 
        ( 1.29735808E-04 ) * va*va*va*va*ppwv + 
        ( 1.29064870E-06 ) * ta*va*va*va*va*ppwv + 
        ( -2.28558686E-06 ) * va*va*va*va*va*ppwv + 
        ( -3.69476348E-02 ) * delta_tmrt*ppwv + 
        ( 1.62325322E-03 ) * ta*delta_tmrt*ppwv + 
        ( -3.14279680E-05 ) * ta*ta*delta_tmrt*ppwv + 
        ( 2.59835559E-06 ) * ta*ta*ta*delta_tmrt*ppwv + 
        ( -4.77136523E-08 ) * ta*ta*ta*ta*delta_tmrt*ppwv + 
        ( 8.64203390E-03 ) * va*delta_tmrt*ppwv + 
        ( -6.87405181E-04 ) * ta*va*delta_tmrt*ppwv + 
        ( -9.13863872E-06 ) * ta*ta*va*delta_tmrt*ppwv + 
        ( 5.15916806E-07 ) * ta*ta*ta*va*delta_tmrt*ppwv + 
        ( -3.59217476E-05 ) * va*va*delta_tmrt*ppwv + 
        ( 3.28696511E-05 ) * ta*va*va*delta_tmrt*ppwv + 
        ( -7.10542454E-07 ) * ta*ta*va*va*delta_tmrt*ppwv + 
        ( -1.24382300E-05 ) * va*va*va*delta_tmrt*ppwv + 
        ( -7.38584400E-09 ) * ta*va*va*va*delta_tmrt*ppwv + 
        ( 2.20609296E-07 ) * va*va*va*va*delta_tmrt*ppwv + 
        ( -7.32469180E-04 ) * delta_tmrt*delta_tmrt*ppwv + 
        ( -1.87381964E-05 ) * ta*delta_tmrt*delta_tmrt*ppwv + 
        ( 4.80925239E-06 ) * ta*ta*delta_tmrt*delta_tmrt*ppwv + 
        ( -8.75492040E-08 ) * ta*ta*ta*delta_tmrt*delta_tmrt*ppwv + 
        ( 2.77862930E-05 ) * va*delta_tmrt*delta_tmrt*ppwv + 
        ( -5.06004592E-06 ) * ta*va*delta_tmrt*delta_tmrt*ppwv + 
        ( 1.14325367E-07 ) * ta*ta*va*delta_tmrt*delta_tmrt*ppwv + 
        ( 2.53016723E-06 ) * va*va*delta_tmrt*delta_tmrt*ppwv + 
        ( -1.72857035E-08 ) * ta*va*va*delta_tmrt*delta_tmrt*ppwv + 
        ( -3.95079398E-08 ) * va*va*va*delta_tmrt*delta_tmrt*ppwv + 
        ( -3.59413173E-07 ) * delta_tmrt*delta_tmrt*delta_tmrt*ppwv + 
        ( 7.04388046E-07 ) * ta*delta_tmrt*delta_tmrt*delta_tmrt*ppwv + 
        ( -1.89309167E-08 ) * ta*ta*delta_tmrt*delta_tmrt*delta_tmrt*ppwv + 
        ( -4.79768731E-07 ) * va*delta_tmrt*delta_tmrt*delta_tmrt*ppwv + 
        ( 7.96079978E-09 ) * ta*va*delta_tmrt*delta_tmrt*delta_tmrt*ppwv + 
        ( 1.62897058E-09 ) * va*va*delta_tmrt*delta_tmrt*delta_tmrt*ppwv + 
        ( 3.94367674E-08 ) * delta_tmrt*delta_tmrt*delta_tmrt*delta_tmrt*ppwv + 
        ( -1.18566247E-09 ) * ta*delta_tmrt*delta_tmrt*delta_tmrt*delta_tmrt*ppwv + 
        ( 3.34678041E-10 ) * va*delta_tmrt*delta_tmrt*delta_tmrt*delta_tmrt*ppwv + 
        ( -1.15606447E-10 ) * delta_tmrt*delta_tmrt*delta_tmrt*delta_tmrt*delta_tmrt*ppwv + 
        ( -2.80626406 ) * ppwv*ppwv + 
        ( 5.48712484E-01 ) * ta*ppwv*ppwv + 
        ( -3.99428410E-03 ) * ta*ta*ppwv*ppwv + 
        ( -9.54009191E-04 ) * ta*ta*ta*ppwv*ppwv + 
        ( 1.93090978E-05 ) * ta*ta*ta*ta*ppwv*ppwv + 
        ( -3.08806365E-01 ) * va*ppwv*ppwv + 
        ( 1.16952364E-02 ) * ta*va*ppwv*ppwv + 
        ( 4.95271903E-04 ) * ta*ta*va*ppwv*ppwv + 
        ( -1.90710882E-05 ) * ta*ta*ta*va*ppwv*ppwv + 
        ( 2.10787756E-03 ) * va*va*ppwv*ppwv + 
        ( -6.98445738E-04 ) * ta*va*va*ppwv*ppwv + 
        ( 2.30109073E-05 ) * ta*ta*va*va*ppwv*ppwv + 
        ( 4.17856590E-04 ) * va*va*va*ppwv*ppwv + 
        ( -1.27043871E-05 ) * ta*va*va*va*ppwv*ppwv + 
        ( -3.04620472E-06 ) * va*va*va*va*ppwv*ppwv + 
        ( 5.14507424E-02 ) * delta_tmrt*ppwv*ppwv + 
        ( -4.32510997E-03 ) * ta*delta_tmrt*ppwv*ppwv + 
        ( 8.99281156E-05 ) * ta*ta*delta_tmrt*ppwv*ppwv + 
        ( -7.14663943E-07 ) * ta*ta*ta*delta_tmrt*ppwv*ppwv + 
        ( -2.66016305E-04 ) * va*delta_tmrt*ppwv*ppwv + 
        ( 2.63789586E-04 ) * ta*va*delta_tmrt*ppwv*ppwv + 
        ( -7.01199003E-06 ) * ta*ta*va*delta_tmrt*ppwv*ppwv + 
        ( -1.06823306E-04 ) * va*va*delta_tmrt*ppwv*ppwv + 
        ( 3.61341136E-06 ) * ta*va*va*delta_tmrt*ppwv*ppwv + 
        ( 2.29748967E-07 ) * va*va*va*delta_tmrt*ppwv*ppwv + 
        ( 3.04788893E-04 ) * delta_tmrt*delta_tmrt*ppwv*ppwv + 
        ( -6.42070836E-05 ) * ta*delta_tmrt*delta_tmrt*ppwv*ppwv + 
        ( 1.16257971E-06 ) * ta*ta*delta_tmrt*delta_tmrt*ppwv*ppwv + 
        ( 7.68023384E-06 ) * va*delta_tmrt*delta_tmrt*ppwv*ppwv + 
        ( -5.47446896E-07 ) * ta*va*delta_tmrt*delta_tmrt*ppwv*ppwv + 
        ( -3.59937910E-08 ) * va*va*delta_tmrt*delta_tmrt*ppwv*ppwv + 
        ( -4.36497725E-06 ) * delta_tmrt*delta_tmrt*delta_tmrt*ppwv*ppwv + 
        ( 1.68737969E-07 ) * ta*delta_tmrt*delta_tmrt*delta_tmrt*ppwv*ppwv + 
        ( 2.67489271E-08 ) * va*delta_tmrt*delta_tmrt*delta_tmrt*ppwv*ppwv + 
        ( 3.23926897E-09 ) * delta_tmrt*delta_tmrt*delta_tmrt*delta_tmrt*ppwv*ppwv + 
        ( -3.53874123E-02 ) * ppwv*ppwv*ppwv + 
        ( -2.21201190E-01 ) * ta*ppwv*ppwv*ppwv + 
        ( 1.55126038E-02 ) * ta*ta*ppwv*ppwv*ppwv + 
        ( -2.63917279E-04 ) * ta*ta*ta*ppwv*ppwv*ppwv + 
        ( 4.53433455E-02 ) * va*ppwv*ppwv*ppwv + 
        ( -4.32943862E-03 ) * ta*va*ppwv*ppwv*ppwv + 
        ( 1.45389826E-04 ) * ta*ta*va*ppwv*ppwv*ppwv + 
        ( 2.17508610E-04 ) * va*va*ppwv*ppwv*ppwv + 
        ( -6.66724702E-05 ) * ta*va*va*ppwv*ppwv*ppwv + 
        ( 3.33217140E-05 ) * va*va*va*ppwv*ppwv*ppwv + 
        ( -2.26921615E-03 ) * delta_tmrt*ppwv*ppwv*ppwv + 
        ( 3.80261982E-04 ) * ta*delta_tmrt*ppwv*ppwv*ppwv + 
        ( -5.45314314E-09 ) * ta*ta*delta_tmrt*ppwv*ppwv*ppwv + 
        ( -7.96355448E-04 ) * va*delta_tmrt*ppwv*ppwv*ppwv + 
        ( 2.53458034E-05 ) * ta*va*delta_tmrt*ppwv*ppwv*ppwv + 
        ( -6.31223658E-06 ) * va*va*delta_tmrt*ppwv*ppwv*ppwv + 
        ( 3.02122035E-04 ) * delta_tmrt*delta_tmrt*ppwv*ppwv*ppwv + 
        ( -4.77403547E-06 ) * ta*delta_tmrt*delta_tmrt*ppwv*ppwv*ppwv + 
        ( 1.73825715E-06 ) * va*delta_tmrt*delta_tmrt*ppwv*ppwv*ppwv + 
        ( -4.09087898E-07 ) * delta_tmrt*delta_tmrt*delta_tmrt*ppwv*ppwv*ppwv + 
        ( 6.14155345E-01 ) * ppwv*ppwv*ppwv*ppwv + 
        ( -6.16755931E-02 ) * ta*ppwv*ppwv*ppwv*ppwv + 
        ( 1.33374846E-03 ) * ta*ta*ppwv*ppwv*ppwv*ppwv + 
        ( 3.55375387E-03 ) * va*ppwv*ppwv*ppwv*ppwv + 
        ( -5.13027851E-04 ) * ta*va*ppwv*ppwv*ppwv*ppwv + 
        ( 1.02449757E-04 ) * va*va*ppwv*ppwv*ppwv*ppwv + 
        ( -1.48526421E-03 ) * delta_tmrt*ppwv*ppwv*ppwv*ppwv + 
        ( -4.11469183E-05 ) * ta*delta_tmrt*ppwv*ppwv*ppwv*ppwv + 
        ( -6.80434415E-06 ) * va*delta_tmrt*ppwv*ppwv*ppwv*ppwv + 
        ( -9.77675906E-06 ) * delta_tmrt*delta_tmrt*ppwv*ppwv*ppwv*ppwv + 
        ( 8.82773108E-02 ) * ppwv*ppwv*ppwv*ppwv*ppwv + 
        ( -3.01859306E-03 ) * ta*ppwv*ppwv*ppwv*ppwv*ppwv + 
        ( 1.04452989E-03 ) * va*ppwv*ppwv*ppwv*ppwv*ppwv + 
        ( 2.47090539E-04 ) * delta_tmrt*ppwv*ppwv*ppwv*ppwv*ppwv + 
        ( 1.48348065E-03 ) * ppwv*ppwv*ppwv*ppwv*ppwv*ppwv 
    )
    return result
