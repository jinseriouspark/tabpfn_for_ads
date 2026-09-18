.PHONY: setup test demo lint mcp clean

PY ?= .venv/bin/python

setup:            ## Create the virtualenv and install everything (offline-capable)
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -e ".[client,llm,mcp,dev]"

test:             ## Run the test suite (no network, no keys needed)
	$(PY) -m pytest -q

demo:             ## Both demos -> reports/ and reports_threads/
	$(PY) -m adlift.cli demo --spec ads --out reports
	$(PY) -m adlift.cli demo --spec threads --out reports_threads

lint:
	$(PY) -m ruff check src tests mcp_server

mcp:              ## Start the MCP server on stdio
	$(PY) -m mcp_server.server

clean:
	rm -rf reports/*.png reports/results.json .pytest_cache .ruff_cache
