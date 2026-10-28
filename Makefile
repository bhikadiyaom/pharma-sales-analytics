# Reproducible pipeline.  On Windows without GNU make, run the python commands directly (see README).
PY ?= python

.PHONY: setup data train analyze app test lint notebooks report all clean

setup:            ## install dependencies
	$(PY) -m pip install -r requirements.txt

data:             ## ingest, validate, clean -> data/processed + SQLite + data-quality report
	$(PY) -m src.pipeline data

train:            ## rolling-origin CV, model selection, forecasts, SHAP, ABC-XYZ, clusters, anomalies, back-test
	$(PY) -m src.pipeline train
	$(PY) -m src.pipeline analyze

app:              ## launch the Streamlit dashboard
	$(PY) -m streamlit run app/Home.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src app tests
	$(PY) -m black --check src app tests

notebooks:        ## (re)build and execute the three notebooks
	$(PY) notebooks/build_notebooks.py

report:           ## fill docs/project_report.md from the result files
	$(PY) -m src.reporting.build_report

all: data train report

clean:
	rm -rf data/processed/* reports/results/* reports/figures/*
