Climate forcing
===============

An incomplete toolbox of scripts and modules used for analysis of climate models and climate data. 

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

- Zelinka, M. D., Andrews, T., Forster, P. M., and Taylor, K. E. (2014), Quantifying components of aerosol‐cloud‐radiation interactions in climate models, J. Geophys. Res. Atmos., 119, 7599–7615, https://doi.org/10.1002/2014JD021710.
- Taylor, K. E., Crucifix, M., Braconnot, P., Hewitt, C. D., Doutriaux, C., Broccoli, A. J., Mitchell, J. F. B., & Webb, M. J. (2007). Estimating Shortwave Radiative Forcing and Response in Climate Models, Journal of Climate, 20(11), 2530–2543, https://doi.org/10.1175/JCLI4143.1

atmos: general atmospheric physics tools
----------------------------------------
humidity: Conversions for specific to relative humidity and vice versa. 

twolayermodel: two-layer energy balance climate model
-----------------------------------------------------
Implementation of the Held et al (2010) and Geoffroy et al (2013a, 2013b) two-layer climate model. Thanks to `Glen Harris <https://www.metoffice.gov.uk/research/people/glen-harris/>`_ for the original code.

- Held, I. M., Winton, M., Takahashi, K., Delworth, T., Zeng, F., & Vallis, G. K. (2010), Probing the Fast and Slow Components of Global Warming by Returning Abruptly to Preindustrial Forcing, J. Climate, 23(9), 2418–2427, https://doi.org/10.1175/2009JCLI3466.1
- Geoffroy, O., Saint-Martin, D., Olivié, D. J. L., Voldoire, A., Bellon, G., & Tytéca, S. (2013a). Transient Climate Response in a Two-Layer Energy-Balance Model. Part I: Analytical Solution and Parameter Calibration Using CMIP5 AOGCM Experiments, J. Climate, 26(6), 1841-1857, https://doi.org/10.1175/JCLI-D-12-00195.1
- Geoffroy, O., Saint-Martin, D., Bellon, G., Voldoire, A., Olivié, D. J. L., & Tytéca, S. (2013b), Transient Climate Response in a Two-Layer Energy-Balance Model. Part II: Representation of the Efficacy of Deep-Ocean Heat Uptake and Validation for CMIP5 AOGCMs, J. Climate, 26(6), 1859-1876, https://doi.org/10.1175/JCLI-D-12-00196.1
- Palmer, M. D., Harris, G. R. and Gregory, J. M. (2018), Extending CMIP5 projections of global mean temperature change and sea level rise due to the thermal expansion using a physically-based emulator, Environ. Res. Lett., 13(8), 084003, https://doi.org/10.1088/1748-9326/aad2e4


utci: Universal Climate Thermal Index
-------------------------------------
Calculates a measure of heat stress based on meteorological data. The code provided is a Python translation of the original FORTRAN, used under kind permission of Peter Bröde. If you use this code please cite:

- Bröde P, Fiala D, Blazejczyk K, Holmér I, Jendritzky G, Kampmann B, Tinz B, Havenith G, 2012. Deriving the operational procedure for the Universal Thermal Climate Index (UTCI). International Journal of Biometeorology 56, 481-494, https://doi.org/10.1007/s00484-011-0454-1
