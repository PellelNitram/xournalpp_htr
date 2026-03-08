docs:
	mkdocs build --clean

tests-installation:
	uv run pytest -v -k "installation"

tests-all:
	uv run pytest -v --durations=0

tests-not-slow:
	uv run pytest -v --durations=0 -m "not slow"

run-pre-commit-hooks:
	pre-commit run --all-files

.PHONY: docs tests-installation tests-all tests-not-slow run-pre-commit-hooks
