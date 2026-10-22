PY ?= python

.PHONY: install smoke test budget generate pointwise replay position run compare

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

position:
	$(PY) -m judge.cli position

replay:
	$(PY) -m judge.cli replay --results results/pointwise_A.json

run:
	$(PY) -m judge.cli run --config A B

compare:
	$(PY) -m judge.cli compare --a A --b B
