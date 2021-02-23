Climate forcing
===============

Some tools that I use in analysis of climate models.

Installation
============

pypi
----

.. code-block::

    pip install climateforcing

development version
-------------------

I strongly recommend doing this inside a virtual environment, e.g. conda, to keep your base python installation clean.

Clone the repository, ``cd`` to ``climateforcing`` and run

.. code-block::

    pip install -e .[dev]


Contents
========

aprp: Approximate Partial Radiative Perturbation
------------------------------------------------
Generates the components of shortwave effective radiative forcing (ERF) from changes in absorption, scattering and cloud amount. For aerosols, this can be used to approximate the ERF from aerosol-radiation interactions (ERFari) and aerosol-cloud interactions (ERFaci). Citations:

- Zelinka, M. D., Andrews, T., Forster, P. M., and Taylor, K. E. (2014), Quantifying components of aerosol‐cloud‐radiation interactions in climate models, J. Geophys. Res. Atmos., 119, 7599– 7615, https://doi.org/10.1002/2014JD021710.
- Taylor, K. E., Crucifix, M., Braconnot, P., Hewitt, C. D., Doutriaux, C., Broccoli, A. J., Mitchell, J. F. B., & Webb, M. J. (2007). Estimating Shortwave Radiative Forcing and Response in Climate Models, Journal of Climate, 20(11), 2530-2543, https://doi.org/10.1175/JCLI4143.1

atmos: general atmospheric physics tools
----------------------------------------
humidity: Conversions for specific to relative humidity and vice versa. 

utci: Universal Climate Thermal Index
-------------------------------------
Calculates a measure of heat stress based on meteorological data. The code provided is a Python translation of the original FORTRAN, used under kind permission of Peter Bröde. If you use this code please cite:

- Bröde P, Fiala D, Blazejczyk K, Holmér I, Jendritzky G, Kampmann B, Tinz B, Havenith G, 2012. Deriving the operational procedure for the Universal Thermal Climate Index (UTCI). International Journal of Biometeorology 56, 481-494, https://doi.org/10.1007/s00484-011-0454-1
