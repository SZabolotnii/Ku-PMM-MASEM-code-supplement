.PHONY: test reproduce verify

test:
	python -m pytest -q

reproduce:
	python -m experiments.run_all

verify: test reproduce
