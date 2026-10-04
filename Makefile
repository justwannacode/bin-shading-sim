# PY=uv run (default) runs inside the uv-managed venv; use `make <target> PY=` in an activated venv.
PY ?= uv run
CONFIG ?= configs/default.yaml

.PHONY: install prepare eda run ablation all test lint format smoke clean

install:
	uv sync --all-extras

prepare:
	$(PY) bidshade prepare --config $(CONFIG)

eda:
	$(PY) bidshade eda --config $(CONFIG)

run:
	$(PY) bidshade run --config $(CONFIG)

# Same experiment without weekday: test days are weekdays never seen in train.
ablation:
	$(PY) bidshade run --config $(CONFIG) --drop-features weekday --tag no_weekday

all: prepare run

test:
	$(PY) pytest

lint:
	$(PY) ruff check src tests

format:
	$(PY) ruff check --fix src tests
	$(PY) ruff format src tests

smoke:
	$(PY) bidshade synthetic --out data/synthetic
	$(MAKE) all CONFIG=configs/synthetic.yaml PY="$(PY)"

clean:
	rm -rf artifacts reports/figures/*.png reports/results.md
