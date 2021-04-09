import os

import pytest

from climateforcing.utils import check_and_download, mkdir_p


def test_mkdir_p():
    mkdir_p("tests/testdata/dummy")
    os.rmdir("tests/testdata/dummy")


def test_mkdir_p_raises():
    # should be no error if tries to create existing dir
    mkdir_p("tests/")
    # should be an error if tries to create a dir over an existing file
    with pytest.raises(OSError):
        mkdir_p("tests/test_utils.py")


def test_check_and_download():
    check_and_download(
        "http://homepages.see.leeds.ac.uk/~mencsm/images/ta_q_kernel.png",
        "tests/testdata/test_image.png",
    )
    check_and_download(
        "http://homepages.see.leeds.ac.uk/~mencsm/images/ta_q_kernel.png",
        "tests/testdata/test_image.png",
        clobber=False,
    )
    check_and_download(
        "http://homepages.see.leeds.ac.uk/~mencsm/images/ta_q_kernel.png",
        "tests/testdata/test_image.png",
        clobber=True,
    )
    os.remove("tests/testdata/test_image.png")
