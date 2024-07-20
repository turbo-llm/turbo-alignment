PYTHON ?= python3

CODE = tests turbo_alignment tutorials

JOBS = 8
MAX_LINE_LENGTH = 119

.PHONY: lint pretty


tests-%:
	coverage run -m pytest -vvs --tb=native $${CI:+--junitxml=report-$(subst tests-,,$@).xml} tests/$(subst tests-,,$@)/
	mv .coverage .coverage.$(subst tests-,,$@)


tests: tests-unit tests-integration
	coverage combine
	[ -n $$CI ] && coverage xml -i || true # always success
	coverage report -i

lint: black pylint mypy

black:
	black --target-version py310 --check --skip-string-normalization --line-length $(MAX_LINE_LENGTH) $(CODE)

pylint:
	pylint --jobs $(JOBS) --rcfile=setup.cfg $${CI:+--output-format=pylint_gitlab.GitlabCodeClimateReporter --output=codeclimate-pylint.json} $(CODE)

mypy:
	mypy --config-file mypy.ini $(CODE) --show-traceback --install-types

pretty:
ifneq ($(CODE),)
	black --target-version py310 --skip-string-normalization --line-length $(MAX_LINE_LENGTH) $(CODE)
	isort $(CODE)
endif

lock:
	rm poetry.lock
	poetry lock

clear:
	rm -f test_*_answers.jsonl
	rm -rf test_*_output
	