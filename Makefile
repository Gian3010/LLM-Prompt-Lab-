.PHONY: install data baseline test notebooks

install:
	pip install -r requirements.txt

data:
	python scripts/generate_dataset.py --n 150

baseline: data
	python scripts/run_experiments.py --models baseline

test:
	python -m pytest -q

notebooks: data
	python scripts/build_notebooks.py
	jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
