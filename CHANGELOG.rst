Changelog
=========

Following the recommendations of https://keepachangelog.com/en/1.0.0/

- ``Added`` for new features.
- ``Changed`` for changes in existing functionality.
- ``Deprecated`` for soon-to-be removed features.
- ``Removed`` for now removed features.
- ``Fixed`` for any bug fixes.
- ``Security`` in case of vulnerabilities.

master
------

v0.2.2
------
- fixed: tentatively, the superfluous "v" from the version string has been dropped from conda releases (`#25 <https://github.com/chrisroadmap/climateforcing/pull/25>`_)
- added: surface pressure introduced into `utci` (`#24 <https://github.com/chrisroadmap/climateforcing/pull/24>`_)
- changed: humidity saturation vapour pressure back to Alduchov and Eskridge (1996) (`#24 <https://github.com/chrisroadmap/climateforcing/pull/24>`_)

v0.2.1
------
- fixed: humidity calculation in `utci` (`#21 <https://github.com/chrisroadmap/climateforcing/pull/21>`_)
- added: conda channel installation setup (`#20 <https://github.com/chrisroadmap/climateforcing/pull/20>`_)
- changed: transposed the output from `cos_mean_solar_zenith_angle` and renamed `utci` function with UTCI expressed in Kelvin (`#19 <https://github.com/chrisroadmap/climateforcing/pull/19>`_)

v0.2.0
------
- added: `solar` module (`#14 <https://github.com/chrisroadmap/climateforcing/pull/14>`_)
- fixed: `SECPERYEAR` for `twolayermodel` now uses tropical year length of 365.24219 days instead of 365 (`#16 <https://github.com/chrisroadmap/climateforcing/pull/16>`_)
- changed: all docstrings converted to numpy style (`#15 <https://github.com/chrisroadmap/climateforcing/pull/15>`_)

v0.1.1
------
- added: `utils` module (`#12 <https://github.com/chrisroadmap/climateforcing/pull/12>`_)
- added: `geometry` module (`#10 <https://github.com/chrisroadmap/climateforcing/pull/10>`_)
- removed: masked array keyword in `humidity` (`#10 <https://github.com/chrisroadmap/climateforcing/pull/10>`_)
- fixed: wrong global mean latitude weighting in `aprp` (`#10 <https://github.com/chrisroadmap/climateforcing/pull/10>`_).

v0.1.0
------
- added: `twolayermodel` (`#7 <https://github.com/chrisroadmap/climateforcing/pull/7>`_).

v0.0.2
------
- fixed: incorrect dependency for `netCDF4` in `setup.py`

v0.0.1
------
- added: `readme-renderer` in `setup.py` to check that readme file is correctly deployed on PyPI 

v0.0.0
------
- added: UTCI
