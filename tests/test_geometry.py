import numpy as np
import pytest

from climateforcing.geometry import global_mean


def test_array_1d():
    lat = np.array([60, 0, -60])
    array = np.array([2, 1, 3])
    global_mean(array, lat)


# is there a more pythonic way than writing out the same test four times?
def test_geometry_raises():
    array = np.array([2, 1, 3])
    lat = np.array([60, 0, -60, -90])
    with pytest.raises(ValueError):
        global_mean(array, lat)
    lat = np.array([[60, 0, -60], [0, 0, 0]])
    with pytest.raises(ValueError):
        global_mean(array, lat)
    lat = np.array([-60, 0, 100])
    with pytest.raises(ValueError):
        global_mean(array, lat)
    lat = np.array([-60, 0, -60])
    with pytest.raises(ValueError):
        global_mean(array, lat)


def test_swap_lat_order():
    array1 = np.array([[2, 1, 3], [4, 8, 1]])
    lat1 = np.array([-60, 0, 60])
    result1 = global_mean(array1, lat1, axis=1)
    array2 = np.array([[3, 1, 2], [1, 8, 4]])
    lat2 = np.array([60, 0, -60])
    result2 = global_mean(array2, lat2, axis=1)
    assert result1 == result2
