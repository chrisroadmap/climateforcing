import pickle

import numpy as np

from climateforcing.twolayermodel import TwoLayerModel

# TODO: add in unit tests


def test_zero():
    scm = TwoLayerModel(
        extforce=np.zeros(500),
        exttime=np.arange(500),
        tbeg=1750,
        tend=2250,
        lamg=4.0 / 3.0,
        t2x=None,
        eff=1.29,
        cmix=6,
        cdeep=75,
        gamma_2l=0.7,
        outtime=np.arange(1750.5, 2250),
        dt=1,
    )
    result = scm.run()

    assert np.allclose(result.hflux, 0)
    assert np.allclose(result.lam_eff, 4 / 3 + (1.29 - 1) * 0.7)
    # Geoffroy et al. 2013 part 2, eq. 28
    assert np.allclose(result.ohc, 0)
    assert np.allclose(result.qtot, 0)
    assert np.allclose(result.tg, 0)
    assert np.allclose(result.tlev, 0)


def test_twolayermodel():
    with open("tests/testdata/twolayermodel.pkl", "rb") as f:
        EXPECTED_RESULT = pickle.load(f)

    scm = TwoLayerModel(
        extforce=4.0 * np.ones(270),
        exttime=np.arange(270),
        tbeg=1750,
        tend=2020,
        lamg=4.0 / 3.0,
        t2x=None,
        eff=1.29,
        cmix=6,
        cdeep=75,
        gamma_2l=0.7,
        outtime=np.arange(1750.5, 2020),
        dt=1,
    )
    result = scm.run()

    assert np.allclose(result.hflux, EXPECTED_RESULT.hflux)
    assert np.allclose(result.lam_eff, EXPECTED_RESULT.lam_eff)
    assert np.allclose(result.ohc, EXPECTED_RESULT.ohc)
    assert np.allclose(result.qtot, EXPECTED_RESULT.qtot)
    assert np.allclose(result.tg, EXPECTED_RESULT.tg)
    assert np.allclose(result.tlev, EXPECTED_RESULT.tlev)

    # Change secperyear and verify OHC is now different
    scm365 = TwoLayerModel(scm_in=result, secperyear=365 * 24 * 60 * 60)
    result365 = scm365.run()
    assert ~np.allclose(result365.ohc, EXPECTED_RESULT.ohc)
