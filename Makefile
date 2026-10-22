PY ?= python

.PHONY: install smoke test budget generate pointwise replay

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

pointwise:
	$(PY) -m judge.cli pointwise --config A

replay:
	$(PY) -m judge.cli replay --results results/pointwise_A.json
