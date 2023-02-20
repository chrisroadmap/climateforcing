import os
import shutil

from climateforcing.utils import check_and_download, mkdir_p


def test_mkdir_p():
    mkdir_p("tests/testdata/dummy")
    os.rmdir("tests/testdata/dummy")


def test_check_and_download():
    # checking both the operation of check_and_download and the capability to
    # make a new directory on the fly
    check_and_download(
        "https://docs.fairmodel.net/en/latest/_images/dimensions.png",
        "tests/newdirectory/test_image.png",
    )
    # clobber working?
    check_and_download(
        "https://docs.fairmodel.net/en/latest/_images/dimensions.png",
        "tests/newdirectory/test_image.png",
        clobber=True,
    )
    # directory specification working?
    check_and_download(
        "https://docs.fairmodel.net/en/latest/_images/dimensions.png",
        "tests/newdirectory",
        clobber=True,
    )
    shutil.rmtree("tests/newdirectory")
