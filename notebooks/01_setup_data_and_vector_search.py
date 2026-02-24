# Databricks notebook source
# This notebook is the "setup" entrypoint:
# - Creates demo Delta tables from the repo CSVs
# - Validates that the configured Vector Search endpoint is ONLINE
# - Ensures a Delta Sync Vector Search index exists on the mappings table

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch pyyaml
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys
import importlib.util

from databricks.vector_search.client import VectorSearchClient

# COMMAND ----------

# Notebook parameters (jobs can override these)
dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")
dbutils.widgets.text("table_prefix", "")
dbutils.widgets.text("rules_table", "")  # Optional override (fully qualified)
dbutils.widgets.text("mappings_table", "")  # Optional override (fully qualified)
dbutils.widgets.text("vs_endpoint_name", "")  # Must be set in config.yaml or here
dbutils.widgets.text("vs_index_name", "")  # Optional override (fully qualified or short name)
dbutils.widgets.text("embedding_model_endpoint", "")

override_catalog = dbutils.widgets.get("catalog").strip() or None
override_schema = dbutils.widgets.get("schema").strip() or None
override_table_prefix = dbutils.widgets.get("table_prefix").strip() or None
override_rules_table = dbutils.widgets.get("rules_table").strip() or None
override_mappings_table = dbutils.widgets.get("mappings_table").strip() or None
override_vs_endpoint = dbutils.widgets.get("vs_endpoint_name").strip() or None
override_vs_index = dbutils.widgets.get("vs_index_name").strip() or None
override_embedding_model = dbutils.widgets.get("embedding_model_endpoint").strip() or None

# COMMAND ----------

# Add repo `src/` to PYTHONPATH so imports work in Databricks jobs
nb_path = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .notebookPath()
    .get()
)
if not nb_path.startswith("/Workspace/"):
    nb_path = f"/Workspace{nb_path}"
parts = nb_path.split("/")
repo_root_ws = "/".join(parts[: parts.index("notebooks")]) if "notebooks" in parts else "/".join(parts[:-1])

# Bundle-deployed code lives under workspace "files/", which may not be on the driver filesystem.
# If `column_mapping` isn't importable, copy workspace `src/` into DBFS and import from there.
if importlib.util.find_spec("column_mapping") is None:
    ws_src = f"file:{repo_root_ws}/src"
    dbfs_dst = "dbfs:/tmp/column-mapping-src"
    try:
        dbutils.fs.rm(dbfs_dst, True)
    except Exception:
        pass
    dbutils.fs.cp(ws_src, dbfs_dst, recurse=True)
    local_src = "/dbfs/tmp/column-mapping-src"
    if local_src not in sys.path:
        sys.path.insert(0, local_src)

from column_mapping.config import compute_effective_config, load_repo_config
from column_mapping.demo_data import ensure_demo_tables
from column_mapping.vector_search import ensure_delta_sync_index, validate_endpoint, wait_for_index_ready
from column_mapping.workspace_paths import get_repo_paths

paths = get_repo_paths(dbutils)

# COMMAND ----------

repo_cfg = load_repo_config(dbutils, repo_root_ws=repo_root_ws)
cfg = compute_effective_config(
    config=repo_cfg,
    catalog=override_catalog,
    schema=override_schema,
    table_prefix=override_table_prefix,
    rules_table=override_rules_table,
    mappings_table=override_mappings_table,
    vs_endpoint_name=override_vs_endpoint,
    vs_index_name_or_full=override_vs_index,
    embedding_model_endpoint=override_embedding_model,
)

print("Creating demo Delta tables from CSVs...")
tables = ensure_demo_tables(
    spark=spark,
    paths=paths,
    catalog=cfg.catalog,
    schema=cfg.schema,
    table_prefix=cfg.table_prefix,
    rules_table=cfg.rules_table,
    mappings_table=cfg.mappings_table,
)

print("Demo tables created:")
for k, v in tables.items():
    print(f"- {k}: {v}")

# Delta Sync Vector Search indexes require Change Data Feed on the source Delta table.
spark.sql(
    f"ALTER TABLE {cfg.mappings_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)
print(f"Enabled Change Data Feed on {cfg.mappings_table}")

# COMMAND ----------

vsc = VectorSearchClient()

# Validate that the configured endpoint exists and is ONLINE.
# This will raise immediately if the endpoint is blank or not found.
endpoint_name = validate_endpoint(vsc=vsc, endpoint_name=cfg.vs_endpoint_name)

print(f"Using Vector Search endpoint: {endpoint_name}")

# COMMAND ----------

index = ensure_delta_sync_index(
    vsc=vsc,
    endpoint_name=endpoint_name,
    index_full_name=cfg.vs_index_full_name,
    source_table_full_name=cfg.mappings_table,
    primary_key=cfg.vs_primary_key,
    embedding_source_column=cfg.vs_embedding_source_column,
    embedding_model_endpoint_name=cfg.embedding_model_endpoint,
    pipeline_type=cfg.vs_pipeline_type,
)

status = wait_for_index_ready(index=index)
print("Vector Search index ready.")
print(f"- index: {cfg.vs_index_full_name}")
print(f"- source_table: {cfg.mappings_table}")
print(f"- rows_indexed: {status.get('num_rows', 'n/a')}")
