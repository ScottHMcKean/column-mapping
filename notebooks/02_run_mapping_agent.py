# Databricks notebook source
# Batch mapping agent: scans Unity Catalog for source columns, runs the
# BM25 + LLM pipeline to propose canonical matches, and writes proposals
# to Delta tables for steward review in the Streamlit app.

# COMMAND ----------

# MAGIC %pip install pyyaml rank-bm25
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys
import importlib.util
import uuid

# COMMAND ----------

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
if importlib.util.find_spec("column_mapping") is None:
    ws_src = repo_root_ws + "/src"
    if ws_src not in sys.path:
        sys.path.insert(0, ws_src)

# COMMAND ----------

import yaml

cfg_path = repo_root_ws + "/config.yaml"
try:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
except FileNotFoundError:
    with open(cfg_path.replace("/Workspace", "")) as f:
        cfg = yaml.safe_load(f)

db = cfg.get("databricks", {})
tables = cfg.get("tables", {})
platforms = cfg.get("platforms", [])
CATALOG = db.get("catalog", "column_mapping")
SCHEMA = db.get("schema", "mapping")
LLM_ENDPOINT = cfg.get("llm", {}).get("endpoint", "databricks-claude-sonnet-4-5")

agent_cfg = cfg.get("agent", {})
CONFIDENCE_HIGH_MIN = agent_cfg.get("confidence_high_min", 85)
CONFIDENCE_MEDIUM_MIN = agent_cfg.get("confidence_medium_min", 60)
MAX_RATIONALE_LEN = agent_cfg.get("max_rationale_length", 500)


def fqn(key):
    return f"{CATALOG}.{SCHEMA}.{tables.get(key, key)}"


T_CANONICAL = fqn("canonical_fields")
T_SOURCE = fqn("source_columns")
T_PROPOSALS = fqn("mapping_proposals")
T_APPROVED = fqn("approved_mappings")
T_AUDIT = fqn("audit_log")
T_RULES = fqn("standardization_rules")

print(f"Catalog: {CATALOG}.{SCHEMA}")
print(f"LLM: {LLM_ENDPOINT}")
print(f"Platforms: {len(platforms)}")

# COMMAND ----------

# Step 1: Discover source columns from INFORMATION_SCHEMA

BATCH_ID = uuid.uuid4().hex[:12]


def spark_sql_fn(query):
    rows = spark.sql(query).collect()
    if not rows:
        return []
    return [r.asDict() for r in rows]


existing_columns = spark_sql_fn(f"SELECT column_id, platform_id, source_table, column_name FROM {T_SOURCE}")
existing_set = {(r["platform_id"], r["source_table"], r["column_name"]) for r in existing_columns}

new_columns = []
for plat in platforms:
    pid = plat["id"]
    src_catalog = plat.get("source_catalog", CATALOG)
    src_schema = plat.get("source_schema", pid)
    try:
        cols = spark_sql_fn(
            f"SELECT table_catalog, table_schema, table_name, column_name, data_type "
            f"FROM {src_catalog}.information_schema.columns "
            f"WHERE table_schema = '{src_schema}'"
        )
    except Exception as exc:
        print(f"  Skipping {pid}: {exc}")
        continue

    for col in cols:
        fq_table = f"{col['table_catalog']}.{col['table_schema']}.{col['table_name']}"
        key = (pid, fq_table, col["column_name"])
        if key not in existing_set:
            cid = uuid.uuid4().hex[:12]
            new_columns.append({
                "column_id": cid,
                "platform_id": pid,
                "source_table": fq_table,
                "column_name": col["column_name"],
                "data_type": col.get("data_type", "STRING"),
                "batch_id": BATCH_ID,
            })
            existing_set.add(key)

print(f"Discovered {len(new_columns)} new columns across {len(platforms)} platforms")

# COMMAND ----------

# Step 2: Write new columns to source_columns


def esc(val):
    if val is None:
        return "NULL"
    return "'" + str(val).replace("'", "''") + "'"


for col in new_columns:
    spark.sql(
        f"INSERT INTO {T_SOURCE} "
        f"(column_id, platform_id, source_table, column_name, data_type, batch_id, detected_at) "
        f"VALUES ({esc(col['column_id'])}, {esc(col['platform_id'])}, "
        f"{esc(col['source_table'])}, {esc(col['column_name'])}, "
        f"{esc(col['data_type'])}, {esc(BATCH_ID)}, current_timestamp())"
    )

print(f"Wrote {len(new_columns)} new source columns.")

# COMMAND ----------

# Step 3: Load context and identify unmapped columns

from column_mapping.mapping_agent import run_mapping_agent

all_source_columns = spark_sql_fn(f"SELECT * FROM {T_SOURCE}")
approved_mappings = spark_sql_fn(f"SELECT * FROM {T_APPROVED}")
canonical_fields = spark_sql_fn(f"SELECT * FROM {T_CANONICAL}")
rules = spark_sql_fn(f"SELECT * FROM {T_RULES}")

mapped_column_ids = {m["column_id"] for m in approved_mappings}
already_proposed_ids = {
    r["column_id"]
    for r in spark_sql_fn(f"SELECT column_id FROM {T_PROPOSALS} WHERE status IN ('pending_review', 'approved')")
}

unmapped = [
    c for c in all_source_columns
    if c["column_id"] not in mapped_column_ids
    and c["column_id"] not in already_proposed_ids
]

plat_names = {p["id"]: p["name"] for p in platforms}

print(f"Unmapped columns needing proposals: {len(unmapped)}")
print(f"Approved mappings (cross-platform context): {len(approved_mappings)}")
print(f"Canonical fields: {len(canonical_fields)}")
print(f"Rules: {len(rules)}")

# COMMAND ----------

# Step 4: Run the mapping agent on unmapped columns

results = []
for i, col in enumerate(unmapped):
    print(f"[{i+1}/{len(unmapped)}] {col['platform_id']} | {col['column_name']}")

    result = run_mapping_agent(
        column_name=col["column_name"],
        platform_id=col["platform_id"],
        platform_name=plat_names.get(col["platform_id"], col["platform_id"]),
        source_columns=all_source_columns,
        approved_mappings=approved_mappings,
        canonical_fields=canonical_fields,
        rules=rules,
        sql_fn=spark_sql_fn,
        llm_endpoint=LLM_ENDPOINT,
    )

    results.append({"column": col, "result": result})

    if result.error:
        print(f"  ERROR: {result.error}")
    else:
        print(f"  -> {result.recommended_canonical_name} ({result.confidence}%)")

# COMMAND ----------

# Step 5: Write proposals to mapping_proposals

written = 0
for item in results:
    col = item["column"]
    res = item["result"]

    confidence_level = (
        "high" if res.confidence >= CONFIDENCE_HIGH_MIN
        else "medium" if res.confidence >= CONFIDENCE_MEDIUM_MIN
        else "low"
    )

    prop_id = uuid.uuid4().hex[:12]
    spark.sql(
        f"INSERT INTO {T_PROPOSALS} "
        f"(proposal_id, column_id, suggested_canonical_id, suggested_canonical_name, "
        f"confidence, confidence_level, reasoning, agent_model, batch_id, status, created_at) "
        f"VALUES ({esc(prop_id)}, {esc(col['column_id'])}, "
        f"{esc(res.recommended_canonical_id)}, {esc(res.recommended_canonical_name)}, "
        f"{res.confidence}, {esc(confidence_level)}, "
        f"{esc((res.rationale or '')[:MAX_RATIONALE_LEN])}, {esc(LLM_ENDPOINT)}, "
        f"{esc(BATCH_ID)}, 'pending_review', current_timestamp())"
    )

    spark.sql(
        f"INSERT INTO {T_AUDIT} "
        f"(event_id, entity_type, entity_id, action, actor, details, created_at) "
        f"VALUES ({esc(uuid.uuid4().hex[:12])}, 'proposal', {esc(prop_id)}, "
        f"'created', 'agent_batch', "
        f"{esc('{\"column\": \"' + col['column_name'] + '\", \"confidence\": ' + str(res.confidence) + '}')}, "
        f"current_timestamp())"
    )

    written += 1

print(f"Wrote {written} mapping proposals.")

# COMMAND ----------

# Summary

summary_df = spark.sql(f"""
    SELECT
        s.platform_id,
        COUNT(DISTINCT s.column_id) AS total_columns,
        COUNT(DISTINCT a.column_id) AS mapped_columns,
        COUNT(DISTINCT p.column_id) AS pending_proposals
    FROM {T_SOURCE} s
    LEFT JOIN {T_APPROVED} a ON a.column_id = s.column_id
    LEFT JOIN {T_PROPOSALS} p ON p.column_id = s.column_id AND p.status = 'pending_review'
    GROUP BY s.platform_id
    ORDER BY s.platform_id
""")
display(summary_df)
