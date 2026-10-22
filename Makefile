PY ?= python

.PHONY: install smoke test

install:
	$(PY) -m pip install -e ".[dev]"

smoke:
	$(PY) -m judge.smoke

test:
	$(PY) -m pytest -q
