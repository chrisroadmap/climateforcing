import os.path

from setuptools import find_packages, setup
from setuptools.command.test import test as TestCommand

import versioneer

PACKAGE_NAME = "climateforcing"
AUTHORS = [
    ("Chris Smith", "c.j.smith1@leeds.ac.uk"),
]
URL = "https://github.com/chrisroadmap/climateforcing"

DESCRIPTION = "Climate-related tools that I use in my work, gathered in a single module"
README = "README.rst"

SOURCE_DIR = "src"

REQUIREMENTS = [
    "numpy",
    "netcdf4"
]
REQUIREMENTS_TESTS = [
    "codecov",
    "coverage",
    "pytest-cov",
    "pytest>=4.0",
]
REQUIREMENTS_DEPLOY = ["twine>=1.11.0", "setuptools>=41.2", "wheel>=0.31.0"]

REQUIREMENTS_DEV = [
    *[
        "bandit",
        "black==19.10b0",
        "flake8",
        "isort>5",
        "pydocstyle",
        "pylint>=2.4.4",
    ],
    *REQUIREMENTS_DEPLOY,
    *REQUIREMENTS_TESTS,
]

REQUIREMENTS_EXTRAS = {
    "deploy": REQUIREMENTS_DEPLOY,
    "dev": REQUIREMENTS_DEV,
    "tests": REQUIREMENTS_TESTS,
}

# no tests/docs in `src` so don't need exclude
PACKAGES = find_packages(SOURCE_DIR)
PACKAGE_DIR = {"": SOURCE_DIR}
PACKAGE_DATA = {"openscm_runner": [os.path.join("adapters", "fair_adapter", "*.csv")]}

# Get the long description from the README file
with open(README, "r") as f:
    README_LINES = ["climateforcing", "==============", ""]
    add_line = False
    for line in f:
        if line.strip() == ".. sec-begin-long-description":
            add_line = True
        elif line.strip() == ".. sec-end-long-description":
            break
        elif add_line:
            README_LINES.append(line.strip())

if len(README_LINES) < 3:
    raise RuntimeError("Insufficient description given")


setup(
    name=PACKAGE_NAME,
    version=versioneer.get_version(),
    description=DESCRIPTION,
    long_description="\n".join(README_LINES),
    long_description_content_type="text/x-rst",
    author=", ".join([author[0] for author in AUTHORS]),
    author_email=", ".join([author[1] for author in AUTHORS]),
    url=URL,
    license="3-Clause BSD License",
    classifiers=[  # full list at https://pypi.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    keywords=["climate", "tools"],
    packages=PACKAGES,
    package_dir=PACKAGE_DIR,
    package_data=PACKAGE_DATA,
    include_package_data=True,
    install_requires=REQUIREMENTS,
    extras_require=REQUIREMENTS_EXTRAS,
    cmdclass=versioneer.get_cmdclass(),
)