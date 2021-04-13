"""Module for calculating the instantanous and time-mean solar zenith angle.

Example
-------
>>> import cftime
>>> from climateforcing.solar.solar_position import modified_julian_date, \
    cos_mean_solar_zenith_angle
>>> import numpy as np
>>> jdate = modified_julian_date(cftime.datetime(2021, 4, 6, 14, 52))
>>> latitude = np.arange(-90, 90.1, 5)
>>> longitude = np.arange(0, 360, 10)
>>> cosz, lit = cos_mean_solar_zenith_angle(jdate, 3, latitude, longitude)

`modified_julian_date` also supports any calendar accepted by `cftime` and makes its
best guess as to what the climate model intended.

Notes
-----
Note that the application of this module is designed for rapid computation of the
time-mean solar zenith angle for climate model applications. No correction for
atmospheric refraction or Earth curvature is given - we care about the incident
radiation at the top of the atmosphere, and it is up to individual climate models to
deal with the radiative transfer for the rest of the journey from the top of atmosphere
to the Earth's surface. If you are looking for something from an Earth-based
perspective, you might be better with `pysolar`, specifically designed for solar
energy use cases. If you want instantaneous zenith angles, you should probably use
`pysolar`, but a function is included:

>>> cosz = cos_solar_zenith(jdate, latitude, longitude)

Note further that these solar zenith angle calculations are not model-specific.
Without access to individual climate model codes, it is impossible to know exactly how
they are implemented in each model. Annoyingly very few models give solar zenith angles
or rsdt at sub-daily time steps, so it is difficult to verify or reverse-engineer the
time-mean zenith angle in different climate models.

What is provided here is a best guess using as accurate as possible almanac data for
the solar position. Leap seconds are not accounted for, so it is possible that by 2021,
this code may already be about 10-20 seconds "fast" compared to reality. We'll assume
this level of detail is not critical for climate model applications and is not likely
to be the largest difference when compared to individual models.

Use at your own risk, etc., etc.

References
----------
.. [1] Reda, I. & Andreas, S. (2004), Solar position algorithm for solar radiation
    applications, Solar Energy, 76, 577-589, doi.org/10.1016/j.solener.2003.12.003.

.. [2] pysolar, https://github.com/pingswept/pysolar

.. [3] Hogan, R. J., and Hirahara, S. (2016), Effect of solar zenith angle specification
    in models on mean shortwave fluxes and stratospheric temperatures, Geophys. Res.
    Lett., 43, 482-488, doi.org/10.1002/2015GL066868.

.. [4] UK Met Office Documentation Paper 23
"""

from .solar_position import (  # noqa: F401
    cos_mean_solar_zenith_angle,
    cos_solar_zenith_angle,
    modified_julian_date,
)
