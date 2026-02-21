# column-mapping
Agentic column mapping between disparate systems of record using:

- Vector Search to retrieve similar prior mappings (RAG context)
- A rules table to enforce naming conventions and standard abbreviations
- An LLM call (`ai_query`) to propose standardized column names + metadata

This repo is designed to be easy to explain and extend. The core flow is split into two notebooks:

- `notebooks/01_setup_data_and_vector_search.py`: loads demo data and creates a Vector Search index
- `notebooks/02_run_column_mapping_agentic.py`: runs the mapping workflow and appends results

## Repo layout

- `data/`: CSVs for a small end-to-end demo (rules, prior mappings, and two source tables)
- `notebooks/`: the primary runnable entrypoints on Databricks
- `src/column_mapping/`: small reusable utilities used by the notebooks
- `databricks.yml` + `resources/`: Databricks Asset Bundle to deploy and run jobs

## Run on Databricks (Databricks Asset Bundle)

Prerequisites:

- You have permissions to create tables in the target `catalog.schema`
- Vector Search is enabled in the workspace
- The model endpoint referenced by `llm_endpoint` is accessible and `ai_query` is available

### Deploy

From a machine with the Databricks CLI configured:

```bash
databricks bundle deploy
```

### Run setup (data + Vector Search)

This loads the demo CSVs into Delta tables and creates/uses a Vector Search endpoint + index.
If `vs_endpoint_name` is not provided, it will prefer an already-online endpoint. If none exist,
it will create `column_mapping_vs_endpoint` by default.

```bash
databricks bundle run column_mapping_setup_vector_search
```

### Run mapping (agentic workflow)

This discovers tables matching `table_prefix` (default `silver*`) and standardizes every column:
Vector Search (similar mappings) + rules -> `ai_query` -> append to the mappings table.

```bash
databricks bundle run column_mapping_run_agentic
```

## Configuration (`config.yaml`)

Default behavior is controlled by `config.yaml` in the repo root (catalog/schema/table names, Vector Search, LLM).
Databricks job parameters (notebook widgets) act as **optional overrides** on top of this config.

## Use your own tables and rules

Both notebooks are parameterized (Databricks widgets). The most important knobs are:

- **Input tables**: `catalog`, `schema`, `table_prefix` (discovers `table_prefix*` tables)
- **Rules table**: `rules_table` (defaults to `${catalog}.${schema}.governance_standardization_rules`)
- **Mappings table**: `mappings_table` (defaults to `${catalog}.${schema}.governance_standardization_mappings`)
- **Vector Search**: `vs_endpoint_name`, `vs_index_name`, `embedding_model_endpoint`
- **LLM**: `llm_endpoint`

## Extensibility (next steps)

The mapping notebook intentionally makes the workflow steps explicit so you can add tools later
(e.g., schema-aware validators, domain classifiers, approval workflows) without rewriting the flow.

## Tests

Run unit tests locally:

```bash
uv run pytest
```
