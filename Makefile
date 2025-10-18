docs:
	mkdocs build --clean

tests-installation:
	pytest -v -k "installation"

run-pre-commit-hooks:
	pre-commit run --all-files

.PHONY: docs tests-installation
