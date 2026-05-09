.PHONY: test lint fmt build clean

# Run all tests
test:
	cd kitchen && uv run pytest tests/ -v --cov=kitchen
	cd recipes && uv run pytest tests/ -v --cov=recipes

# Lint both packages
lint:
	cd kitchen && uv run ruff check .
	cd recipes && uv run ruff check .

# Format both packages
fmt:
	cd kitchen && uv run ruff format .
	cd recipes && uv run ruff format .

# Build distributable wheels and sdists
build:
	python -m build kitchen/ --outdir dist/
	python -m build recipes/ --outdir dist/
	twine check dist/*

clean:
	rm -rf dist/ kitchen/.pytest_cache recipes/.pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
