# ============================================================
# FootPredict-Pro Makefile
# ============================================================

.PHONY: help install fetch-data fetch-csv train backtest predict app test clean

LEAGUE ?= 39
SEASONS ?= 2022 2023
HOME ?= "Manchester City"
AWAY ?= "Arsenal"

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install all dependencies
	pip install -r requirements.txt

fetch-data:  ## Fetch match data from API-Football (LEAGUE=39, SEASONS="2022 2023")
	@for season in $(SEASONS); do \
		python src/data_ingestion/api_football.py --league $(LEAGUE) --season $$season; \
	done

fetch-csv:  ## Fetch data from football-data.co.uk CSV fallback (LEAGUE=E0)
	python src/data_ingestion/football_data_co.py --league $(LEAGUE)

fetch-understat:  ## Fetch xG data from Understat (LEAGUE=EPL)
	python src/data_ingestion/understat.py --league $(LEAGUE) --seasons "$(SEASONS)"

train:  ## Train all models (LEAGUE=39)
	python src/training/train.py --league $(LEAGUE) --seasons "$(SEASONS)"

train-all:  ## Train on all configured leagues
	python src/training/train.py --all-leagues

backtest:  ## Run temporal backtesting (SEASONS="2021 2022 2023")
	python src/training/backtest.py --seasons "$(SEASONS)"

predict:  ## Quick prediction (HOME="Team A" AWAY="Team B")
	python src/inference/predict_match.py --home $(HOME) --away $(AWAY)

app:  ## Launch Streamlit web UI
	streamlit run app/streamlit_app.py

api:  ## Launch FastAPI server
	uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

test:  ## Run test suite with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing

test-fast:  ## Run tests (no coverage)
	pytest tests/ -v -x

clean:  ## Clean generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete 2>/dev/null; \
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null; \
	echo "Cleaned."

clean-data:  ## Remove downloaded data (careful!)
	rm -rf data/raw/* data/processed/*

clean-models:  ## Remove trained models (careful!)
	rm -rf models/*.joblib models/*.pkl
