# used openscm-runner as a template
# https://github.com/openscm/openscm-runner

.DEFAULT_GOAL := help

VENV_DIR ?= .venv

TESTS_DIR=./tests

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

.PHONY: help
help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

checks: $(VENV_DIR)  ## run all the checks
	@echo "\n\n=== black ==="; $(VENV_DIR)/bin/black --check src tests setup.py --exclude climateforcing/_version.py || echo "--- black failed ---" >&2; \
		echo "\n\n=== flake8 ==="; $(VENV_DIR)/bin/flake8 src tests setup.py || echo "--- flake8 failed ---" >&2; \
		echo "\n\n=== isort ==="; $(VENV_DIR)/bin/isort --check-only --quiet src tests setup.py || echo "--- isort failed ---" >&2; \
		echo "\n\n=== pydocstyle ==="; $(VENV_DIR)/bin/pydocstyle src || echo "--- pydocstyle failed ---" >&2; \
		echo "\n\n=== pylint ==="; $(VENV_DIR)/bin/pylint src || echo "--- pylint failed ---" >&2; \
		echo "\n\n=== notebook tests ==="; $(VENV_DIR)/bin/pytest notebooks -r a --nbval --sanitize-with $(NOTEBOOKS_SANITIZE_FILE) || echo "--- notebook tests failed ---" >&2; \
		echo "\n\n=== tests ==="; $(VENV_DIR)/bin/pytest tests -r a --cov=openscm_runner --cov-report='' \
			&& $(VENV_DIR)/bin/coverage report --fail-under=95 || echo "--- tests failed ---" >&2; \
		echo

.PHONY: format
format:  ## re-format files
	make isort
	make black

black: $(VENV_DIR)  ## apply black formatter to source and tests
	$(VENV_DIR)/bin/black --exclude _version.py setup.py src tests; 

isort: $(VENV_DIR)  ## format the code
	$(VENV_DIR)/bin/isort src tests setup.py; 

pylint: $(VENV_DIR)  ## apply linter
	$(VENV_DIR)/bin/pylint src;

docs: $(VENV_DIR)  ## build the docs
	$(VENV_DIR)/bin/sphinx-build -M html docs/source docs/build

test:  $(VENV_DIR) ## run the full testsuite
	$(VENV_DIR)/bin/pytest tests --cov -rfsxEX --cov-report term-missing

virtual-environment:  ## update venv, create a new venv if it doesn't exist
	make $(VENV_DIR)

$(VENV_DIR): setup.py
	[ -d $(VENV_DIR) ] || python3 -m venv $(VENV_DIR)

	$(VENV_DIR)/bin/pip install --upgrade pip wheel
	$(VENV_DIR)/bin/pip install -e .[dev]
	touch $(VENV_DIR)

first-venv: ## create a new virtual environment for the very first repo setup
	python3 -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install versioneer
	# don't touch here as we don't want this venv to persist anyway
