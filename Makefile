docs:
	mkdocs build --clean

tests-installation:
	pytest -v -k "installation"

tests-all:
	pytest -v --durations=0

tests-not-slow:
	pytest -v --durations=0 -m "not slow"

run-pre-commit-hooks:
	pre-commit run --all-files

.PHONY: docs tests-installation tests-all tests-not-slow run-pre-commit-hooks
