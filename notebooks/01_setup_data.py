# Databricks notebook source
# Setup: loads seed CSVs into Unity Catalog and creates metadata tables.
#
# Run this once (or to reset) before running 02_run_mapping_agent.

# COMMAND ----------

# MAGIC %pip install pyyaml
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import yaml
import importlib.util
import sys

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

cfg_path = repo_root_ws + "/config.yaml"
try:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
except FileNotFoundError:
    with open(cfg_path.replace("/Workspace", "")) as f:
        cfg = yaml.safe_load(f)

db = cfg.get("databricks", {})
tables_cfg = cfg.get("tables", {})
platforms = cfg.get("platforms", [])
CATALOG = db.get("catalog", "column_mapping")
SCHEMA = db.get("schema", "mapping")


def fqn(key):
    return f"{CATALOG}.{SCHEMA}.{tables_cfg.get(key, key)}"


print(f"Catalog: {CATALOG}")
print(f"Schema: {SCHEMA}")
print(f"Platforms: {len(platforms)}")

# COMMAND ----------

# Step 1: Create catalog, main schema, and platform schemas

gold_cfg = cfg.get("gold", {})
GOLD_CATALOG = gold_cfg.get("catalog", CATALOG)
GOLD_SCHEMA = gold_cfg.get("schema", "gold")

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {GOLD_CATALOG}.{GOLD_SCHEMA}")
print(f"  Gold schema: {GOLD_CATALOG}.{GOLD_SCHEMA}")
for plat in platforms:
    src_schema = plat.get("source_schema", plat["id"])
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{src_schema}")
    print(f"  Schema: {CATALOG}.{src_schema}")

# COMMAND ----------

# Step 2: Create metadata tables

T_CANONICAL = fqn("canonical_fields")
T_SOURCE = fqn("source_columns")
T_PROPOSALS = fqn("mapping_proposals")
T_APPROVED = fqn("approved_mappings")
T_AUDIT = fqn("audit_log")
T_RULES = fqn("standardization_rules")

ddls = {
    T_CANONICAL: f"""CREATE TABLE IF NOT EXISTS {T_CANONICAL} (
        canonical_id STRING,
        canonical_name STRING,
        data_type STRING,
        business_definition STRING,
        domain_category STRING,
        is_active BOOLEAN,
        created_by STRING,
        created_at TIMESTAMP,
        updated_at TIMESTAMP
    ) USING DELTA""",
    T_SOURCE: f"""CREATE TABLE IF NOT EXISTS {T_SOURCE} (
        column_id STRING,
        platform_id STRING,
        source_table STRING,
        column_name STRING,
        data_type STRING,
        batch_id STRING,
        detected_at TIMESTAMP
    ) USING DELTA""",
    T_PROPOSALS: f"""CREATE TABLE IF NOT EXISTS {T_PROPOSALS} (
        proposal_id STRING,
        column_id STRING,
        suggested_canonical_id STRING,
        suggested_canonical_name STRING,
        confidence DOUBLE,
        confidence_level STRING,
        reasoning STRING,
        agent_model STRING,
        batch_id STRING,
        status STRING,
        assigned_to STRING,
        created_at TIMESTAMP,
        reviewed_at TIMESTAMP,
        reviewed_by STRING
    ) USING DELTA""",
    T_APPROVED: f"""CREATE TABLE IF NOT EXISTS {T_APPROVED} (
        mapping_id STRING,
        column_id STRING,
        canonical_id STRING,
        proposal_id STRING,
        approved_by STRING,
        approved_at TIMESTAMP
    ) USING DELTA""",
    T_AUDIT: f"""CREATE TABLE IF NOT EXISTS {T_AUDIT} (
        event_id STRING,
        entity_type STRING,
        entity_id STRING,
        action STRING,
        actor STRING,
        details STRING,
        created_at TIMESTAMP
    ) USING DELTA""",
    T_RULES: f"""CREATE TABLE IF NOT EXISTS {T_RULES} (
        rule_id STRING,
        rule_type STRING,
        pattern STRING,
        replacement STRING,
        description STRING,
        is_active BOOLEAN
    ) USING DELTA""",
}

for table_name, ddl in ddls.items():
    spark.sql(ddl)
    print(f"  Table: {table_name}")

# COMMAND ----------

# Step 3: Load canonical_fields.csv and standardization_rules.csv

import pandas as pd
from pyspark.sql.functions import current_timestamp, lit

import os
data_dir = repo_root_ws + "/data"
if not os.path.isdir(data_dir):
    data_dir = repo_root_ws.replace("/Workspace", "") + "/data"

canonical_pdf = pd.read_csv(f"{data_dir}/canonical_fields.csv")
canonical_df = spark.createDataFrame(canonical_pdf)
canonical_df = (
    canonical_df
    .withColumnRenamed("is_locked", "is_active")
    .withColumn("created_at", current_timestamp())
    .withColumn("updated_at", current_timestamp())
)
canonical_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T_CANONICAL)
print(f"  Loaded {canonical_pdf.shape[0]} canonical fields into {T_CANONICAL}")

rules_pdf = pd.read_csv(f"{data_dir}/standardization_rules.csv")
rules_df = spark.createDataFrame(rules_pdf)
rules_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T_RULES)
print(f"  Loaded {rules_pdf.shape[0]} rules into {T_RULES}")

# COMMAND ----------

# Step 4: Load platform CSVs into their respective schemas

for plat in platforms:
    pid = plat["id"]
    csv_file = plat.get("seed_csv")
    table_name = plat.get("seed_table")
    if not csv_file or not table_name:
        print(f"  Skipping {pid}: no seed_csv/seed_table in config.yaml")
        continue

    src_schema = plat.get("source_schema", pid)
    fq_table = f"{CATALOG}.{src_schema}.{table_name}"

    pdf = pd.read_csv(f"{data_dir}/{csv_file}")
    sdf = spark.createDataFrame(pdf)
    sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(fq_table)
    print(f"  {fq_table}: {len(pdf)} rows, {len(pdf.columns)} columns")

# COMMAND ----------

# Step 5: Truncate any stale data from prior runs

for t in [T_SOURCE, T_PROPOSALS, T_APPROVED, T_AUDIT]:
    spark.sql(f"TRUNCATE TABLE {t}")
    print(f"  Truncated {t}")

print("\nSetup complete. Run 02_run_mapping_agent next.")

# COMMAND ----------

# Step 6: Grant app service principal access to the catalog
#
# The Databricks App runs as a service principal that needs
# USE_CATALOG, USE_SCHEMA, SELECT, and MODIFY on the catalog.
# This step looks up the app by name and grants permissions.

APP_NAME = db.get("app_name", "column-mapper")

try:
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    app = w.apps.get(APP_NAME)
    sp_id = app.service_principal_client_id
    spark.sql(
        f"GRANT USE_CATALOG, USE_SCHEMA, CREATE_SCHEMA, SELECT, MODIFY "
        f"ON CATALOG {CATALOG} TO `{sp_id}`"
    )
    print(f"  Granted catalog permissions to app service principal: {sp_id}")
except Exception as e:
    print(f"  Note: Could not grant app permissions (app may not exist yet): {e}")
    print(f"  Re-run this job after 'databricks bundle run column_mapper' creates the app.")
