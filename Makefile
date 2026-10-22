PY ?= python

.PHONY: install smoke test budget generate

install:
	$(PY) -m pip install -e ".[dev]"

smoke:
	$(PY) -m judge.smoke

test:
	$(PY) -m pytest -q

budget:
	$(PY) -m judge.cli budget

generate:
	$(PY) -m judge.cli generate
