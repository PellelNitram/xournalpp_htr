docs:
	uv run mkdocs build --clean

tests-installation:
	uv run pytest -v -k "installation"

tests-all:
	uv run pytest -v --durations=0

tests-not-slow:
	uv run pytest -v --durations=0 -m "not slow"

tests-docker:
	uv run pytest -m slow tests/test_docker.py -v -s --log-cli-level=INFO

run-pre-commit-hooks:
	pre-commit run --all-files

serve-data-annotation-tool:
	uv run python -m http.server 8000 --directory tools/data_annotation_tool

.PHONY: docs tests-installation tests-all tests-not-slow run-pre-commit-hooks serve-data-annotation-tool
