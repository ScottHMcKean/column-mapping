# Databricks notebook source
# This notebook runs the mapping workflow:
# - Discovers input tables (by prefix) under a catalog + schema
# - For each column in each table, uses Vector Search to retrieve similar prior mappings
# - Loads active rules
# - Calls an LLM via ai_query to propose a standardized column name + metadata
# - Appends results to the mappings table (which can then be indexed / re-synced)

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch pyyaml
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys
from typing import List

from databricks.vector_search.client import VectorSearchClient
from pyspark.sql.functions import col, current_timestamp, lit, monotonically_increasing_id

# COMMAND ----------

# Notebook parameters (jobs can override these)
dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")
dbutils.widgets.text("table_prefix", "")
dbutils.widgets.text("rules_table", "")  # Optional override (fully qualified)
dbutils.widgets.text("mappings_table", "")  # Optional override (fully qualified)
dbutils.widgets.text("vs_endpoint_name", "")  # Optional preferred name; otherwise best active
dbutils.widgets.text("vs_index_name", "")  # Optional override (fully qualified or short name)
dbutils.widgets.text("llm_endpoint", "")
dbutils.widgets.text("top_k", "")

override_catalog = dbutils.widgets.get("catalog").strip() or None
override_schema = dbutils.widgets.get("schema").strip() or None
override_table_prefix = dbutils.widgets.get("table_prefix").strip() or None
override_rules_table = dbutils.widgets.get("rules_table").strip() or None
override_mappings_table = dbutils.widgets.get("mappings_table").strip() or None
override_vs_endpoint = dbutils.widgets.get("vs_endpoint_name").strip() or None
override_vs_index = dbutils.widgets.get("vs_index_name").strip() or None
override_llm_endpoint = dbutils.widgets.get("llm_endpoint").strip() or None
override_top_k = dbutils.widgets.get("top_k").strip() or None

# COMMAND ----------

# Add repo `src/` to PYTHONPATH so imports work in Databricks jobs
nb_path = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .notebookPath()
    .get()
)
parts = nb_path.split("/")
repo_root_ws = "/".join(parts[: parts.index("notebooks")]) if "notebooks" in parts else "/".join(parts[:-1])
src_path = f"{repo_root_ws}/src"
if src_path not in sys.path:
    sys.path.append(src_path)

from column_mapping.config import compute_effective_config, load_repo_config
from column_mapping.agentic_mapping import (
    discover_managed_tables,
    safe_int_confidence,
    source_system_from_table,
    standardize_column_agentic,
)
from column_mapping.vector_search import ensure_endpoint

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
    llm_endpoint=override_llm_endpoint,
    top_k=int(override_top_k) if override_top_k else None,
)

vsc = VectorSearchClient()
endpoint_name = ensure_endpoint(
    vsc=vsc,
    preferred_name=cfg.vs_endpoint_name,
    create_if_missing=False,
)

index = vsc.get_index(endpoint_name=endpoint_name, index_name=cfg.vs_index_full_name)
idx_status = index.describe()
if not (idx_status.get("status") or {}).get("ready"):
    raise RuntimeError(
        f"Vector Search index is not ready: endpoint={endpoint_name} index={cfg.vs_index_full_name}"
    )

print(f"Using Vector Search endpoint: {endpoint_name}")
print(f"Using Vector Search index: {cfg.vs_index_full_name}")

# COMMAND ----------

tables: List[str] = discover_managed_tables(
    spark=spark, catalog=cfg.catalog, schema=cfg.schema, table_prefix=cfg.table_prefix
)

if not tables:
    raise RuntimeError(
        f"No tables found for catalog={cfg.catalog} schema={cfg.schema} prefix={cfg.table_prefix}"
    )

print("Discovered tables:")
for t in tables:
    print(f"- {cfg.catalog}.{cfg.schema}.{t}")

# COMMAND ----------

all_rows = []

for table_name in tables:
    full_table = f"{cfg.catalog}.{cfg.schema}.{table_name}"
    source_system = source_system_from_table(table_name=table_name, table_prefix=cfg.table_prefix)

    columns = [r.col_name for r in spark.sql(f"DESCRIBE TABLE {full_table}").collect()]

    for column_name in columns:
        result = standardize_column_agentic(
            spark=spark,
            index=index,
            rules_table=cfg.rules_table,
            llm_endpoint=cfg.llm_endpoint,
            column_name=column_name,
            source_system=source_system,
            top_k=cfg.top_k,
        )

        all_rows.append(
            (
                result.get("source_system"),
                result.get("platform_header"),
                result.get("standardized_header"),
                result.get("domain"),
                result.get("data_type"),
                safe_int_confidence(result.get("confidence_score")),
                result.get("reasoning"),
            )
        )

results_df = spark.createDataFrame(
    all_rows,
    [
        "source_system",
        "platform_header",
        "standardized_header",
        "domain",
        "data_type",
        "confidence_score",
        "reasoning",
    ],
)

display(results_df)

# COMMAND ----------

# Conform to the mappings table schema and append.
final_df = (
    results_df.withColumn("approval_status", lit("pending_review"))
    .withColumn("mapping_id", monotonically_increasing_id().cast("string"))
    .withColumn("created_at", current_timestamp())
    .withColumn("transformation_notes", lit("Generated by LLM"))
    .withColumn("approved_by", lit(None).cast("string"))
    .withColumn("approved_at", lit(None).cast("timestamp"))
    .withColumn("confidence_score", col("confidence_score").cast("int"))
    .select(
        "mapping_id",
        "source_system",
        "platform_header",
        "standardized_header",
        "domain",
        "data_type",
        "transformation_notes",
        "confidence_score",
        "approval_status",
        "approved_by",
        "approved_at",
        "created_at",
    )
)

final_df.write.mode("append").option("mergeSchema", "true").saveAsTable(cfg.mappings_table)

print(f"Appended {final_df.count()} mappings to {cfg.mappings_table}")

# Best-effort: trigger a sync for TRIGGERED indexes so new rows become searchable.
try:
    if hasattr(index, "sync"):
        index.sync()
    elif hasattr(vsc, "sync_index"):
        vsc.sync_index(endpoint_name=endpoint_name, index_name=cfg.vs_index_full_name)
    print("Triggered Vector Search index sync.")
except Exception as e:
    print(f"Skipped index sync (not supported or failed): {e}")

