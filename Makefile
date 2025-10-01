SHELL := /bin/bash
PY := python

.PHONY: all data features predict evaluate clean

all: data features predict evaluate

data:
	$(PY) -m src.cli ingest

features:
	$(PY) -m src.cli build-features

predict:
	$(PY) -m src.cli predict

evaluate:
	$(PY) -m src.cli evaluate

clean:
	rm -rf data/features/* data/predictions/* reports/*
