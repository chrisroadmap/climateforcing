from setuptools import find_packages, setup

import versioneer

PACKAGE_NAME = "climateforcing"
AUTHORS = [
    ("Chris Smith", "c.j.smith1@leeds.ac.uk"),
]
URL = "https://github.com/chrisroadmap/climateforcing"

DESCRIPTION = "Climate-related tools that I use in my work, gathered in a single module"
README = "README.rst"

SOURCE_DIR = "src"

REQUIREMENTS = ["cftime>=1.4.0", "numpy", "netCDF4", "pandas", "scipy", "tqdm"]
REQUIREMENTS_TESTS = [
    "codecov",
    "coverage",
    "pytest-cov",
    "pytest>=4.0",
]
REQUIREMENTS_DEPLOY = ["twine>=1.11.0", "setuptools>=41.2", "wheel>=0.31.0"]
REQUIREMENTS_DOCS = ["sphinx>=1.4", "sphinx_rtd_theme"]

REQUIREMENTS_DEV = [
    *[
        "black==19.10b0",
        "flake8",
        "isort>5",
        "pydocstyle",
        "pylint>=2.4.4",
        "readme-renderer",
    ],
    *REQUIREMENTS_DEPLOY,
    *REQUIREMENTS_DOCS,
    *REQUIREMENTS_TESTS,
]

REQUIREMENTS_EXTRAS = {
    "deploy": REQUIREMENTS_DEPLOY,
    "dev": REQUIREMENTS_DEV,
    "docs": REQUIREMENTS_DOCS,
    "tests": REQUIREMENTS_TESTS,
}

# no tests/docs in `src` so don't need exclude
PACKAGES = find_packages(SOURCE_DIR)
PACKAGE_DIR = {"": SOURCE_DIR}

# Get the long description from the README file
with open("README.rst", "r", encoding="utf8") as fh:
    long_description = fh.read()

setup(
    name=PACKAGE_NAME,
    version=versioneer.get_version(),
    description=DESCRIPTION,
    long_description=long_description,
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
        "Programming Language :: Python :: 3.9",
    ],
    keywords=["climate", "tools"],
    packages=PACKAGES,
    package_dir=PACKAGE_DIR,
    include_package_data=True,
    install_requires=REQUIREMENTS,
    extras_require=REQUIREMENTS_EXTRAS,
    cmdclass=versioneer.get_cmdclass(),
)
