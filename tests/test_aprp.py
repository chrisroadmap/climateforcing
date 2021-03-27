import numpy as np
from climateforcing.aprp import aprp, create_input

# TODO: add in unit tests


def test_aprp_access_esm1_5():
    # these values are not expected to be realistic global means as our test data is
    # a small area in the tropics
    EXPECTED_RESULT = {
        "t1": 0.015656651939777565,
        "t2": -0.11131543138394552,
        "t3": 0.03621888328689638,
        "t4": -0.016439486854532722,
        "t5": -0.365565897874444,
        "t6": 0.27068423458611834,
        "t7": -1.760830367218397,
        "t8": 0.11666082099794635,
        "t9": 2.00885025935897,
        "t2_clr": -0.6014480068357444,
        "t3_clr": 0.1495480779271396,
        "ERFariSWclr": -0.4518999467318717,
        "ERFariSW": -0.16997821514240372,
        "ERFaciSW": 0.3646806457861222,
        "albedo": -0.0007828350913690186,
        "ERFariLW": -0.846055215472465,
        "ERFaciLW": -3.297287354345663,
    }
    BASEDIR = "tests/testdata/ACCESS-ESM1-5/piClim-control/"
    PERTDIR = "tests/testdata/ACCESS-ESM1-5/piClim-aer/"
    base, pert, lat = create_input(BASEDIR, PERTDIR, lw=True, latout=True)
    result = aprp(base, pert, lat=lat, lw=True, globalmean=True)
    for key, value in result.items():
        assert np.allclose(value, EXPECTED_RESULT[key])
