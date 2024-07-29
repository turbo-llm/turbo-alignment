PYTHON ?= python3

CODE = tests turbo_alignment

JOBS = 8
MAX_LINE_LENGTH = 119

.PHONY: lint pretty


tests-%:
	WANDB_MODE=offline coverage run -m pytest -vvs --tb=native $${CI:+--junitxml=report-$(subst tests-,,$@).xml} tests/$(subst tests-,,$@)/
	mv .coverage .coverage.$(subst tests-,,$@)


tests: tests-unit tests-integration tests-cli
	coverage combine
	[ -n $$CI ] && coverage xml -i || true # always success
	coverage report -i

lint: black flake pylint mypy

black:
	black --target-version py310 --check --skip-string-normalization --line-length $(MAX_LINE_LENGTH) $(CODE)

flake:
	flake8 --max-line-length 119 --jobs $(JOBS) --statistics $${CI:+--format=gl-codeclimate --output=codeclimate-flake8.json} $(CODE)

pylint:
	pylint --jobs $(JOBS) --rcfile=setup.cfg $${CI:+--output-format=pylint_gitlab.GitlabCodeClimateReporter --output=codeclimate-pylint.json} $(CODE)

mypy:
	mypy --config-file mypy.ini $(CODE) --show-traceback

pretty:
ifneq ($(CODE),)
	black --target-version py310 --skip-string-normalization --line-length $(MAX_LINE_LENGTH) $(CODE) tutorials
	isort $(CODE) tutorials
	unify --in-place --recursive $(CODE) tutorials
endif

lock:
	rm poetry.lock
	poetry lock

clear:
	rm -f test_*_answers.jsonl
	rm -rf test_*_output
	