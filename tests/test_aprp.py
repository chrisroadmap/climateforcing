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
        "albedo": -0.004681652319370494,
        "ERFari_scat": -0.4433192884978622,
        "ERFari_abs": 0.40955245497308285,
        "ERFaci_scat": -1.7441234051275887,
        "ERFaci_abs": 0.14376985338862716,
        "ERFaci_amt": 2.012463079695719,
        "ERFariSW": -0.0337668276752567,
        "ERFaciSW": 0.41210950703083343,
        "ERFariLW": -0.8487686048639489,
        "ERFaciLW": -3.321506047334925,
    }
    BASEDIR = "tests/testdata/ACCESS-ESM1-5/piClim-control/"
    PERTDIR = "tests/testdata/ACCESS-ESM1-5/piClim-aer/"
    base, pert, lat = create_input(BASEDIR, PERTDIR, longwave=True, latout=True)
    result_3d = aprp(base, pert, longwave=True)
    for key, value in result_3d.items():
        result_1d = global_mean(value, lat=lat, axis=1)
        assert np.allclose(result_1d, EXPECTED_RESULT[key], atol=1e-5)


def test_create_input_slice():
    BASEDIR = "tests/testdata/ACCESS-ESM1-5/piClim-control/"
    PERTDIR = "tests/testdata/ACCESS-ESM1-5/piClim-aer/"
    base1, pert1 = create_input(BASEDIR, PERTDIR)
    base2, pert2 = create_input(BASEDIR, PERTDIR, slc=slice(0, 1, None))
    for key in base1:
        assert ~np.allclose(base1[key], base2[key])


def test_create_input_raises():
    BASEDIR = ""
    PERTDIR = ""
    with pytest.raises(RuntimeError):  # should be custom class
        base1, pert1 = create_input(BASEDIR, PERTDIR)
