"""Module for planetary geometry."""

import numpy as np


# next two functions thanks to
# https://stackoverflow.com/questions/4983258/python-how-to-check-list-monotonicity
def _strictly_increasing(array_to_test):
    return all(x < y for x, y in zip(array_to_test, array_to_test[1:]))


def _strictly_decreasing(array_to_test):
    return all(x > y for x, y in zip(array_to_test, array_to_test[1:]))


def global_mean(array, lat, axis=None):
    """Calculate area-weighted mean.

    Quick and dirty method, assumes latitude bounds are halfway (arithmetically)
    between latitude points. If you're being super strict you want to use
    something like `iris` (e.g. this method is not time-aware).

    Parameters
    ----------
        array : array_like
            array to apply the latitude-weighting to. Must be at least 1d.
        lat : array_like
            latitude points of the array in degrees. Should be 1d.
        axis : int or None
            axis to perform the weighting over. None is valid for a 1d `array`,
            otherwise must be specified.

    Returns
    -------
        result : float
            area-weighted mean of array
    """
    # initial checks
    if array.ndim == 1:
        axis = 0
    if lat.ndim != 1:
        raise ValueError("`lat` must be an array of dimension 1")
    if len(lat) != array.shape[axis]:
        raise ValueError("`lat` must be the same length as `array` axis to be meaned")
    if max(lat) > 90 or min(lat) < -90:
        raise ValueError("`lat` must be in the range [-90, 90]")
    # is latitude ascending or descending?
    if _strictly_increasing(lat):
        latbounds = np.concatenate(([-90], 0.5 * (lat[1:] + lat[:-1]), [90]))
        weights = np.diff(np.sin(np.radians(latbounds)))
    elif _strictly_decreasing(lat):
        latbounds = np.concatenate(([90], 0.5 * (lat[1:] + lat[:-1]), [-90]))
        weights = -np.diff(np.sin(np.radians(latbounds)))
    else:
        raise ValueError("`lat` must be strictly increasing or decreasing")
    # First average over latitude axis, then take mean of the result
    result = np.mean(np.average(array, weights=weights, axis=axis))
    return result
