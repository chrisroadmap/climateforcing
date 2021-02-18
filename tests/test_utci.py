from climateforcing.utci import utci
import numpy as np

def test_utci():
    EXPECTED_RESULT = np.array([19.62569676, 21.23458492])
    TEST_RESULT = utci(np.array([295,296], dtype=float), np.array([303,304], dtype=float), np.array([6.0,6.0], dtype=float), np.array([100,100], dtype=float))
    assert np.allclose(TEST_RESULT, EXPECTED_RESULT)
