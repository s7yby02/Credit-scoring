include .env
export
export PYTHONPATH := .
export MONGO_URI := mongodb://localhost:27017
export MLFLOW_TRACKING_URI := http://localhost:5000
export GOOGLE_APPLICATION_CREDENTIALS := $(CURDIR)/secrets/gcp-key.json

PY := .venv/bin/python
DBT := .venv-dbt/bin/dbt

data:        ## training batch: 3,000 applications with known outcomes
	$(PY) generator/generate.py --batch train --users 3000 --start 2024-01-01 --end 2025-09-30
prod:        ## recent applications, same population
	$(PY) generator/generate.py --batch prod --users 800 --last-days 30 --no-outcomes --seed 7
prod-drift:  ## recent applications, shifted population
	$(PY) generator/generate.py --batch prod --users 800 --last-days 30 --no-outcomes --seed 7 --drift
load:
	$(PY) scripts/load_mongo.py data/train
load-prod:
	$(PY) scripts/load_mongo.py data/prod
dbt:
	cd dbt/credit_scoring && ../../$(DBT) build --profiles-dir .
train:
	$(PY) training/train.py
drift:
	$(PY) monitoring/drift.py
test:
	$(PY) -m pytest -q tests
score:
	$(PY) scripts/sample_request.py data/prod 50
