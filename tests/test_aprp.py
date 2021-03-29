import numpy as np
import pytest

from climateforcing.aprp import aprp, cloud_radiative_effect, create_input
from climateforcing.geometry import global_mean


def test_cloud_radiative_effect_raises():
    base = {}
    pert = {}
    with pytest.raises(ValueError):
        cloud_radiative_effect(base, pert)


def test_aprp_raises():
    base = {}
    pert = {}
    with pytest.raises(ValueError):
        aprp(base, pert)
    with pytest.raises(ValueError):
        aprp(base, pert, longwave=True)
    varlist = ["rsdt", "rsus", "rsds", "clt", "rsdscs", "rsuscs", "rsut", "rsutcs"]
    # check different shape input fails
    for var in varlist:
        base[var] = np.zeros((1, 1, 1))
        pert[var] = np.zeros((1, 1, 1))
    base["rsdscs"] = np.zeros((8, 1, 1))
    with pytest.raises(ValueError):
        aprp(base, pert)


# TODO: throw error for invalid values
def test_clt_percent_hit():
    # checks that we hit the cloud fraction divisor statement.
    varlist = ["rsdt", "rsus", "rsds", "clt", "rsdscs", "rsuscs", "rsut", "rsutcs"]
    base = {}
    pert = {}
    for var in varlist:
        base[var] = np.zeros((1, 1, 1))
        pert[var] = np.zeros((1, 1, 1))
    base["clt"] = np.ones((1, 1, 1)) * 105
    pert["clt"] = np.ones((1, 1, 1)) * 105
    aprp(base, pert, clt_percent=True)
    base["clt"] = np.ones((1, 1, 1)) * -1.05
    pert["clt"] = np.ones((1, 1, 1)) * -1.05
    aprp(base, pert, clt_percent=False)


def test_aprp_access_esm1_5():
    # these values are not expected to be realistic global means as our test data is
    # a small area in the tropics
    EXPECTED_RESULT = {
        "t1": 0.015690050209499505,
        "t2": -0.11169482070528963,
        "t3": 0.03628321688570137,
        "t4": -0.016643288519753274,
        "t5": -0.3646235907233642,
        "t6": 0.2702892397012102,
        "t7": -1.7754625968973907,
        "t8": 0.11863184693029566,
        "t9": 2.0124631258387526,
        "t2_clr": -0.6004451771086404,
        "t3_clr": 0.14946570231037004,
        "ERFariSWclr": -0.45097949302816626,
        "ERFariSW": -0.16974595857509908,
        "ERFaciSW": 0.35563230710202826,
        "albedo": -0.0009532384845432344,
        "ERFariLW": -0.8487686048639488,
        "ERFaciLW": -3.3215060473349243,
    }
    BASEDIR = "tests/testdata/ACCESS-ESM1-5/piClim-control/"
    PERTDIR = "tests/testdata/ACCESS-ESM1-5/piClim-aer/"
    base, pert, lat = create_input(BASEDIR, PERTDIR, longwave=True, latout=True)
    result_3d = aprp(base, pert, longwave=True)
    for key, value in result_3d.items():
        result_1d = global_mean(value, lat=lat, axis=1)
        assert np.allclose(result_1d, EXPECTED_RESULT[key])
