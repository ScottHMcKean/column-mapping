# Column Mapper -- Design Document

**Date:** 2026-03-03 (revised 2026-03-17)

---

## The Problem

Every financial platform has its own naming conventions for the same data concepts. `Acct-ID`, `AccountID`, `account_id`, `Account ID` -- they all mean the same thing. Multiply this by hundreds of columns across dozens of platforms during a migration and you have a governance problem that no spreadsheet will solve.

This project takes a hub-and-spoke approach: **discover source columns via batch, propose canonical matches with an AI agent, and let stewards review and approve in a simple two-tab app.** Every machine-generated output is a proposal -- never a fact.

---

## Architecture: Batch / App / Audit

Three clearly separated concerns:

```
BATCH (Scheduled Job)              APP (Streamlit)              AUDIT (Delta Table)
---------------------              ---------------              -------------------
Scan INFORMATION_SCHEMA            Data Mapping Work Queue      Immutable append-only
  for configured platforms           - Review proposals           Written by batch + app
Write to source_columns             - Approve / Reject / Flag    Records: who, what,
Run AI agent (BM25 + LLM)           - Create new entries           when, why
Write to mapping_proposals         Master File Management
Log to audit_log                     - Browse canonical fields
                                     - Drill-down by platform
                                     - Generate gold views
                                   Audit Log
                                     - Filter and browse events
                                   All writes -> audit_log
```

### Batch

The batch process runs as a Databricks notebook or job. It:

1. Scans `INFORMATION_SCHEMA.COLUMNS` for each configured platform (catalog/schema pair)
2. Inserts newly discovered columns into `source_columns`
3. Runs the mapping agent (BM25 keyword search + LLM synthesis) on unmapped columns
4. Writes mapping proposals to `mapping_proposals` with confidence scores and reasoning
5. Logs all actions to `audit_log`

### App

The Streamlit app is read-heavy and write-light:

- Reads from Delta tables via SQL warehouse (cached with TTL)
- Steward actions: approve, reject, flag proposals; create canonical entries
- Generates gold layer SQL views from approved mappings
- Every write also creates an audit_log entry
- The app never runs the AI agent directly -- it only reviews batch-generated proposals

### Audit

- Append-only Delta table
- Written by both batch and app
- Fields: event_id, entity_type, entity_id, action, actor, details (JSON), created_at
- Provides complete chain of custody for regulatory compliance

---

## Data Model

Six Delta tables. All reads and writes go through the serverless SQL warehouse.

### 1. canonical_fields (Master File)

The golden vocabulary. Every approved mapping points to a canonical field. This is the hub of the hub-and-spoke model.

| Column | Type | Purpose |
|---|---|---|
| canonical_id | STRING | Primary key |
| canonical_name | STRING | Standard name (e.g. fund_identifier) |
| data_type | STRING | Expected type |
| business_definition | STRING | What this concept means |
| domain_category | STRING | Business domain |
| is_active | BOOLEAN | Soft delete |
| created_by | STRING | Who defined this concept |
| created_at | TIMESTAMP | When created |
| updated_at | TIMESTAMP | Last update |

### 2. source_columns

Raw columns discovered from source tables. Populated by batch scanning Unity Catalog INFORMATION_SCHEMA.

| Column | Type | Purpose |
|---|---|---|
| column_id | STRING | Primary key |
| platform_id | STRING | Config-defined platform ID |
| source_table | STRING | Fully-qualified table name |
| column_name | STRING | Original column name from the source |
| data_type | STRING | Source data type |
| batch_id | STRING | Which batch discovered this column |
| detected_at | TIMESTAMP | When discovered |

### 3. mapping_proposals

AI-suggested mappings between source columns and canonical fields. Populated by the batch agent. Reviewed by stewards in the app.

| Column | Type | Purpose |
|---|---|---|
| proposal_id | STRING | Primary key |
| column_id | STRING | FK to source_columns |
| suggested_canonical_id | STRING | FK to canonical_fields (null for new suggestions) |
| suggested_canonical_name | STRING | Name for new entry suggestions |
| confidence | DOUBLE | Score 0-100 |
| confidence_level | STRING | high / medium / low |
| reasoning | STRING | Agent's rationale |
| agent_model | STRING | LLM model used |
| batch_id | STRING | Which batch created this proposal |
| status | STRING | pending_review / approved / rejected / flagged |
| assigned_to | STRING | Steward assignment (nullable) |
| created_at | TIMESTAMP | When proposed |
| reviewed_at | TIMESTAMP | When reviewed |
| reviewed_by | STRING | Who reviewed |

### 4. approved_mappings

Steward-confirmed links between source columns and canonical fields. Written exclusively by the app.

| Column | Type | Purpose |
|---|---|---|
| mapping_id | STRING | Primary key |
| column_id | STRING | FK to source_columns |
| canonical_id | STRING | FK to canonical_fields |
| proposal_id | STRING | FK to mapping_proposals |
| approved_by | STRING | Who approved |
| approved_at | TIMESTAMP | When approved |

### 5. audit_log

Immutable record of every action. Required for regulatory compliance.

| Column | Type | Purpose |
|---|---|---|
| event_id | STRING | Primary key |
| entity_type | STRING | canonical / mapping / proposal / rule / gold_views |
| entity_id | STRING | ID of the affected entity |
| action | STRING | created / approved / rejected / flagged / revoked |
| actor | STRING | Who performed the action |
| details | STRING | JSON context |
| created_at | TIMESTAMP | When the action occurred |

### 6. standardization_rules

Abbreviation and naming convention rules used by the batch agent during deterministic standardization.

| Column | Type | Purpose |
|---|---|---|
| rule_id | STRING | Primary key |
| rule_type | STRING | abbreviation (extensible) |
| pattern | STRING | The short form (e.g. acct) |
| replacement | STRING | The standard form (e.g. account) |
| description | STRING | Human-readable explanation |
| is_active | BOOLEAN | Soft delete |

---

## Configuration

All configuration lives in `config.yaml`. No hardcoded table names, platform definitions, or catalog references.

| Section | Keys | Purpose |
|---|---|---|
| databricks | catalog, schema, warehouse_id, profile | Unity Catalog location and compute |
| tables | canonical_fields, source_columns, etc. | Table name overrides |
| platforms | id, name, source_catalog, source_schema | Source systems to scan |
| gold | catalog, schema | Where gold views are created |
| llm | endpoint | Model serving endpoint for ai_query |

Adding a new source system is one config entry. The batch process picks it up automatically.

---

## The App: Two Primary Views

### Data Mapping Work Queue

The steward's inbox. Shows incoming column headers with AI-suggested canonical matches.

- **Stats**: Pending Review, High Priority, Needs Discussion, Completion Rate, Flagged, Active Batches
- **Filters**: Platform, status, free-text search
- **Table**: Suggested Match, Confidence, Header, Platform, Source Table, and one column per platform showing what they already call this concept (cross-platform context)
- **Actions**: Approve Selected, Flag for Discussion, Reject Selected
- **Detail panel**: Reasoning, reassignment to a different canonical field, or creation of a new master entry

### Master File Management

The canonical registry. Shows all standardized master entries with platform coverage.

- **Stats**: Active Master Entries, Total Mappings, Unmapped Headers, Coverage %
- **Filters**: Text search, domain filter
- **Table**: Standardized Name, Business Definition, Platform Coverage, Mapped Headers, Domain, Type, Status
- **Drill-down**: Expand a row to see all mapped source columns grouped by platform with source table, column name, approved by, and date
- **Create Entry**: Form for manually creating new canonical fields (with dedup detection)
- **Gold Views**: Generate and execute CREATE VIEW DDLs for source tables with approved mappings
- **Rules**: Browse and add abbreviation rules used by the batch agent

---

## Gold Layer

Approved mappings are materialized as SQL views in the gold catalog/schema. For each source table with approved mappings, a view is generated that renames source columns to their canonical names:

```sql
CREATE OR REPLACE VIEW gold_catalog.gold_schema.platform__table_name AS
SELECT
    `Bloomberg_ID` AS bloomberg_id,
    `Currency` AS currency_code
FROM source_catalog.source_schema.table_name
```

Stewards preview the SQL and execute with a button click. Each execution is audit-logged.

---

## Deployment

### Local development

```bash
uv run streamlit run app/app.py
```

### Databricks Asset Bundle

```bash
databricks bundle deploy
```

The bundle deploys the batch notebook as a scheduled job and the Streamlit app as a Databricks App.
