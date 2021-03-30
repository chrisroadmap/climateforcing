from climateforcing.atmos.humidity import relative_to_specific, specific_to_relative


def test_relative_to_specific():
    EXPECTED_RESULT = 0.0060003648770542905
    test_result = relative_to_specific(0.5, air_temperature=290, pressure=1e5)
    assert test_result == EXPECTED_RESULT
    test_result = relative_to_specific(
        50, air_temperature=290, pressure=1e5, rh_percent=True
    )
    assert test_result == EXPECTED_RESULT


def test_specific_to_relative():
    EXPECTED_RESULT = 0.49996959542779756
    test_result = specific_to_relative(0.006, air_temperature=290, pressure=1e5)
    assert test_result == EXPECTED_RESULT
    test_result = specific_to_relative(
        0.006, air_temperature=290, pressure=1e5, rh_percent=True
    )
    assert test_result == EXPECTED_RESULT * 100
