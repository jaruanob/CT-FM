### Variables
# Define shell and Python environment variables
SHELL := /usr/bin/env bash
PYTHON := python
PYTHONPATH := `pwd`

# Install
.PHONY: setup
setup: 
	pip install uv
	
#* Installation
.PHONY: install
install:
	uv pip install -e .

#* Formatters
.PHONY: codestyle
codestyle:
	uvx pyupgrade --exit-zero-even-if-changed --py37-plus **/*.py
	uvx isort --settings-path pyproject.toml ./
	uvx black --config pyproject.toml ./

.PHONY: formatting
formatting: codestyle

.PHONY: check-codestyle
check-codestyle:
	uvx isort --diff --check-only --settings-path pyproject.toml ./
	uvx black --diff --check --config pyproject.toml ./
	uv run pylint lighter-ct-fm

.PHONY: bump-prerelease
bump-prerelease:
	uvx --with poetry-bumpversion poetry version prerelease

.PHONY: bump-patch
bump-patch:
	uvx --with poetry-bumpversion poetry version patch

.PHONY: bump-minor
bump-minor:
	uvx --with poetry-bumpversion poetry version minor

.PHONY: bump-major
bump-major:
	uvx --with poetry-bumpversion poetry version major

.PHONY: mypy
mypy:
	uvx mypy --config-file pyproject.toml ./

