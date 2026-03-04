# Column Mapping

### Design Document

**Date:** 2026-03-03

---

## The Problem

Every organization has the same columns scattered across systems under different names. `Acct-ID`, `AccountID`, `account_id`, `Account ID` -- they all mean the same thing. Multiply this by hundreds of columns across dozens of systems and you have a governance nightmare that no amount of documentation will fix.

Pairwise mapping (System A to System B) explodes combinatorially. Documentation rots. Tribal knowledge leaves when people do.

This project takes a different approach: **teach by example**. Build a growing library of approved column mappings and a set of naming rules, then use vector similarity and an LLM to propose standardizations for new columns. A data steward reviews and approves. Every approval makes the system smarter.

---

## Data Philosophy

**1. Examples over configuration.** The best way to teach a naming standard is to show it, not describe it. Every approved mapping becomes a searchable example for future proposals. The system learns your style.

**2. Rules are guardrails, not the engine.** Rules (abbreviation expansions, naming conventions, domain classifications) constrain the LLM's output. But the real intelligence comes from the examples. Rules keep it consistent; examples make it accurate.

**3. Humans decide, machines propose.** The LLM never writes to the approved mapping table without a steward's explicit approval. Every decision is attributed, timestamped, and versioned.

**4. One table, one job.** Each table has a single responsibility. There are no multi-purpose tables. The data model is four tables. That's it.

**5. Feedback loops compound.** Approved mappings feed back into the vector index. The more you use the system, the better its proposals get. Early effort pays exponential dividends.

**6. Every change is a fact.** Mappings use Type 2 slowly-changing dimensions. No row is ever updated in place. Every approval, rejection, and edit inserts a new version row and closes the old one. The full history of every mapping decision is always queryable.

**7. Generality by design.** Nothing in this system is specific to a particular domain, schema, or source system. Point it at any catalog and it works.

---

## Data Objects

There are four core tables. Everything else is derived.

### 1. Canonical Columns

The canonical concepts that all source columns map to. These are your gold-table column definitions -- the single vocabulary for your entire data estate.

| Column | Type | Purpose |
|---|---|---|
| canonical_name | STRING | Primary key. The standard column name (e.g. `customer_id`) |
| business_definition | STRING | What this concept means in plain language |
| domain | STRING | Business domain (`Customer Data`, `Financial`, etc.) |
| expected_data_type | STRING | The expected data type in standardized form |
| created_by | STRING | Who defined this concept |
| created_at | TIMESTAMP | When it was created |

Every `standardized_header` in the mappings table references a `canonical_name` here. This is the hub of a hub-and-spoke model: source columns are the spokes, canonical columns are the hub.

You can answer "which systems have a `customer_id`?" by joining mappings to canonical columns. You can build a cross-system pivot view showing every source column that maps to every canonical concept.

Canonical columns emerge naturally from the mapping process. When a steward approves a mapping with a new `standardized_header` that does not yet exist in the canonical table, it gets created automatically.

### 2. Standardization Rules

The rules of your naming convention. These are the guardrails the LLM must follow.

| Column | Type | Purpose |
|---|---|---|
| rule_id | STRING | Primary key |
| rule_type | STRING | `naming_convention`, `abbreviation`, `domain`, `data_type` |
| rule_key | STRING | The thing being defined (e.g. `acct`, `amt`) |
| rule_value | STRING | The standard form (e.g. `account`, `amt`) |
| rule_description | STRING | Human-readable explanation |
| examples | STRING | Concrete before/after examples |
| is_active | BOOLEAN | Soft delete |

Rule types serve distinct purposes:

- **naming_convention** -- structural rules like "use snake_case" or "remove special characters"
- **abbreviation** -- expansion and contraction mappings (`acct` -> `account`, `mgr` stays `mgr`)
- **domain** -- classification categories (`Customer Data`, `Financial`, `Operational`, `Reference`, `Location`, `Organizational`)
- **data_type** -- suffix-based type inference (`_id` -> STRING, `_amt` -> DECIMAL, `_date` -> DATE)

### 3. Standardization Mappings (Type 2 SCD)

The edges from source columns to canonical concepts. This table uses **Type 2 slowly-changing dimensions** -- no row is ever updated in place. Every state change (proposal, approval, rejection, edit) inserts a new version and closes the old one.

| Column | Type | Purpose |
|---|---|---|
| mapping_id | STRING | Logical identity of the mapping (stable across versions) |
| version | INT | Version number within this mapping_id (1, 2, 3...) |
| source_system | STRING | Where this column lives (e.g. `salesforce`, `shopify`) |
| platform_header | STRING | The raw column name as it appears in the source |
| standardized_header | STRING | The canonical name this maps to (FK to canonical_columns) |
| domain | STRING | Business domain classification |
| data_type | STRING | Standardized data type |
| transformation_notes | STRING | How the column name was transformed |
| confidence_score | INT | 0-100, how confident the proposal was |
| approval_status | STRING | `pending_review`, `approved`, `rejected` |
| valid_from | TIMESTAMP | When this version became the active version |
| valid_to | TIMESTAMP | When this version was superseded (NULL = current) |
| changed_by | STRING | Who created this version |
| change_reason | STRING | Why this version exists |

The composite key is `(mapping_id, version)`. The current state of all mappings is:

```sql
SELECT * FROM standardization_mappings WHERE valid_to IS NULL
```

The full history of a single mapping is:

```sql
SELECT * FROM standardization_mappings
WHERE mapping_id = 'MAP-001'
ORDER BY version
```

Each row is an immutable fact: *at this point in time, this person made this decision about this mapping*. Approved current-version rows are the training data for future proposals. The more approved examples, the better the system gets.

### 4. Source Column Inventory

The physical columns discovered from source systems via `information_schema`. This is a one-way sync -- metadata flows in, never back.

| Column | Type | Purpose |
|---|---|---|
| source_system | STRING | Logical name for the source |
| catalog_name | STRING | Unity Catalog catalog |
| schema_name | STRING | Unity Catalog schema |
| table_name | STRING | Table containing the column |
| column_name | STRING | The raw column name |
| data_type | STRING | Source data type |
| column_comment | STRING | Any existing documentation |
| discovered_at | TIMESTAMP | When the column was found |

This table is populated by scanning `information_schema` for tables matching a configured prefix. It represents the work queue: columns that need standardization.

---

## Data Flow

The system operates as a flywheel with four stages.

```
                    +------------------+
                    |  Source Systems   |
                    | (Unity Catalog)  |
                    +--------+---------+
                             |
                    1. Discover columns
                             |
                             v
                    +------------------+
                    | Column Inventory  |
                    |  (one-way sync)  |
                    +--------+---------+
                             |
                    2. Propose mappings
                      (FAISS + Rules + LLM)
                             |
                             v
                    +------------------+
                    |    Mappings       |
                    | (pending_review) |
                    +--------+---------+
                             |
                    3. Steward reviews
                      (App: approve / reject / edit)
                             |
                             v
                    +------------------+
                    |    Mappings       |
                    |   (approved)     |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
     4a. Feed back into               4b. Manual edits
         FAISS index                   also become approved
         (better future proposals)     examples + can add rules
```

### Stage 1: Discover

Scan `information_schema` for tables matching a configured prefix. Extract column names and metadata. Write to the column inventory. This is a one-way sync -- source metadata flows into the mapping system, never back out.

### Stage 2: Propose

For each undiscovered column:

1. **Vector search** -- Query the FAISS index (built from approved mappings) for the most similar `platform_header` values. Retrieve the top-k matches with their `standardized_header`, `domain`, and `data_type`.
2. **Load rules** -- Pull active rules from the rules table. Format naming conventions, abbreviations, and domain/type guidance into the prompt.
3. **LLM call** -- Send the column name, similar examples, and rules to the LLM. Ask for a JSON response with: `standardized_header`, `domain`, `data_type`, `confidence_score`, `reasoning`.
4. **Write proposal** -- Insert the result into the mappings table with `approval_status = 'pending_review'`.

### Stage 3: Review

Data stewards use the app to review proposals. They can:

- **Approve** -- Accept the LLM's proposal as-is. The mapping becomes an approved example.
- **Reject** -- Decline the proposal. It stays in the table for audit but is excluded from the FAISS index.
- **Edit** -- Change the `standardized_header`, `domain`, or `data_type` via dropdowns before approving. The edited version becomes the approved example.

### Stage 4: Feedback

Approved mappings are loaded into the FAISS index. The next time a similar column appears, the system already knows the answer. Manual edits (stage 3) are particularly valuable -- they correct the LLM's mistakes and teach the system your preferences.

Stewards can also add new rules directly through the app. If they notice the LLM consistently getting an abbreviation wrong, they add a rule. Rules and examples work together: rules constrain, examples demonstrate.

---

## Lineage

### Type 2 SCD on Mappings

Every state change to a mapping produces a new version row. The old row is closed (its `valid_to` is set), and a new row is inserted with `valid_to = NULL`. Nothing is ever updated in place except the `valid_to` timestamp on the row being superseded.

**Lifecycle of a mapping:**

```
Version 1: LLM proposes mapping
  mapping_id=MAP-100, version=1, approval_status=pending_review
  valid_from=2026-01-26 10:00, valid_to=NULL
  changed_by=system, change_reason=LLM proposal

  [Steward approves]

Version 2: Steward approves
  mapping_id=MAP-100, version=1, valid_to=2026-02-04 15:43  (closed)
  mapping_id=MAP-100, version=2, approval_status=approved
  valid_from=2026-02-04 15:43, valid_to=NULL
  changed_by=scotty, change_reason=Approved by steward

  [Steward later edits the domain]

Version 3: Steward edits
  mapping_id=MAP-100, version=2, valid_to=2026-03-01 09:00  (closed)
  mapping_id=MAP-100, version=3, approval_status=approved, domain=Financial
  valid_from=2026-03-01 09:00, valid_to=NULL
  changed_by=scotty, change_reason=Edited: domain changed from Reference to Financial
```

### What you can query

**Current state** -- all mappings as they stand right now:

```sql
SELECT * FROM standardization_mappings WHERE valid_to IS NULL
```

**Point-in-time snapshot** -- what the mappings looked like on a specific date:

```sql
SELECT * FROM standardization_mappings
WHERE valid_from <= '2026-02-01' AND (valid_to IS NULL OR valid_to > '2026-02-01')
```

**Full audit trail** -- every decision ever made about a mapping:

```sql
SELECT * FROM standardization_mappings
WHERE mapping_id = 'MAP-100'
ORDER BY version
```

**Who changed what, when** -- governance report:

```sql
SELECT mapping_id, version, approval_status, changed_by, change_reason, valid_from
FROM standardization_mappings
WHERE changed_by != 'system'
ORDER BY valid_from DESC
```

### Canonical Cross-System View

Because every mapping points to a canonical column, you can pivot across all systems:

```sql
SELECT
    c.canonical_name,
    c.business_definition,
    m.source_system,
    m.platform_header,
    m.data_type
FROM canonical_columns c
LEFT JOIN standardization_mappings m
    ON c.canonical_name = m.standardized_header
    AND m.valid_to IS NULL
    AND m.approval_status = 'approved'
ORDER BY c.canonical_name, m.source_system
```

This answers: *for every canonical concept, which source systems have a column that maps to it, and what do they call it?*

---

## Architecture

### Lakebase Backend

All four tables live in Lakebase (Databricks-managed PostgreSQL). Lakebase provides:

- Low-latency reads for the app
- Transactional writes for Type 2 version inserts
- Standard PostgreSQL interface for the SDK

The app connects to Lakebase via the Databricks SDK, loading tables into memory on startup.

### In-Memory Vector Search (FAISS)

The app does not depend on a running Databricks Vector Search endpoint. Instead:

1. On startup, load current approved mappings from Lakebase (`WHERE valid_to IS NULL AND approval_status = 'approved'`)
2. Vectorize each `platform_header` using TF-IDF character n-grams (2-4 grams). Character n-grams are well-suited for column name matching where patterns like abbreviations, case changes, and separators matter more than semantic meaning.
3. Build a FAISS `IndexFlatIP` index on the L2-normalized TF-IDF vectors
4. Query the index locally when stewards select a mapping to see similar approved examples

This approach is fast (sub-millisecond queries), lightweight (only `faiss-cpu` and `scikit-learn`, no heavy ML frameworks), and scales to tens of thousands of mappings comfortably in memory. The index is rebuilt on each app restart, manual reload, or after committing new approvals.

### Scaling with Databricks Vector Search

FAISS works well for interactive use in the app with up to tens of thousands of mappings. For larger deployments or batch processing, the system can use Databricks Vector Search with Delta Sync:

- **Delta Sync index** on the mappings table automatically keeps the vector index in sync as new approved mappings are appended. Configure with `pipeline_type: TRIGGERED` to control when syncs happen, or `CONTINUOUS` for real-time updates.
- **Managed embeddings** via a Foundation Model endpoint (e.g., `databricks-bge-large-en`) eliminate the need to compute embeddings in the app.
- **Serverless scaling** handles spiky workloads during large batch mapping jobs without provisioning infrastructure.

The batch mapping notebook (`02_run_column_mapping_agentic.py`) already supports Databricks Vector Search. To switch the app from FAISS to Databricks Vector Search, replace the local index build with a call to the Vector Search SDK -- the query interface is the same (text in, scored results out).

When to switch:
- **Stay with FAISS**: fewer than 50,000 approved mappings, single app instance, low latency required
- **Move to Vector Search**: more than 50,000 mappings, multiple consumers, need Delta Sync for continuous updates, or want managed embeddings

### One-Way Sync Pattern

Data flows in one direction through the system:

```
Source information_schema  -->  Column Inventory (read-only)
                                      |
                                      v
Mapping Job (batch)  ---------> Mappings Table (version 1: pending_review)
                                      |
                                      v
App (steward review) ---------> Mappings Table (version N: approved/rejected/edited)
                                      |
                                      v
FAISS Index (in-memory) <------- Mappings Table (current approved rows, loaded on startup)
```

The column inventory is never written back to the source. The mappings table is append-only -- new versions are inserted, old versions have their `valid_to` closed, but no row is ever deleted or modified beyond that single timestamp. The FAISS index is a read-only derivative of current approved mappings, rebuilt from scratch on reload.

---

## The App

The Streamlit app is a visual tool that guides data stewards through the review process. It is not the system -- it is a window into the system.

### What the steward sees

1. **Mapping table** -- Filterable by source system, domain, approval status, and confidence score. Sortable. Multi-select for batch operations.

2. **Metrics** -- Total mappings, average confidence, pending review count, uncommitted edits.

3. **Actions** -- Approve, reject, or edit selected mappings. Commit or discard changes.

4. **Manual edit with dropdowns** -- For any mapping, the steward can override:
   - `standardized_header`: free text input
   - `domain`: dropdown populated from canonical columns
   - `data_type`: dropdown populated from canonical columns

   When a manual edit is approved, it becomes a new approved example in the mappings table, improving future proposals. If the standardized header does not yet exist in the canonical columns table, a new canonical entry is auto-created.

5. **Add rule** -- Stewards can add new abbreviation, naming convention, domain, or data type rules directly from the app via an expandable form. New rules take effect on the next mapping proposal.

6. **Similar mappings** -- When a single mapping is selected, FAISS similarity search shows the most similar approved mappings. This helps stewards see prior decisions for similar column names before making their own.

### How data moves in the app

1. **Load** -- On startup (or manual reload), fetch current mappings (`WHERE valid_to IS NULL`) from Lakebase into a Pandas DataFrame. Build the FAISS index from approved rows.
2. **Filter** -- All filtering happens in-memory on the cached DataFrame. No round-trips to the database for UI interactions.
3. **Stage** -- Approve/reject/edit actions are staged in session state. Nothing is written until the steward commits.
4. **Commit** -- For each staged decision, the app performs a Type 2 version insert:
   - Close the current version: `UPDATE SET valid_to = now() WHERE mapping_id = X AND valid_to IS NULL`
   - Insert a new version row with the updated fields, `version + 1`, `valid_from = now()`, `valid_to = NULL`
   - If the approved `standardized_header` is new, auto-create a canonical column entry
   - Refresh the local DataFrame and FAISS index

---

## Extensibility

This system is designed to be general. Here is how to adapt it.

### Different source systems

Add tables to the configured catalog/schema with the configured prefix. The discovery step finds them automatically. No code changes required.

### Different naming conventions

Edit the rules table. Add abbreviation rules, change domain categories, adjust data type mappings. The LLM reads rules at proposal time, so changes take effect immediately.

### Different LLM

Change the `llm_endpoint` in `config.yaml`. The prompt format is model-agnostic JSON-in/JSON-out.

### Larger scale

For organizations with hundreds of thousands of columns:
- The batch mapping job (notebook) can use Databricks Vector Search with Delta Sync instead of FAISS
- The app's FAISS index can be filtered to only relevant source systems
- The column inventory can be partitioned by catalog or system

### Custom transformation logic

The `transformation_notes` field can be extended to store SQL transformation expressions (e.g., `CAST(order_ts AS DATE)`) for use in downstream harmonized views. This is a natural extension point but is not required for column name standardization.

---

## Project Structure

```
column-mapping/
  app/
    app.py                     # Streamlit review app
    app.yaml                   # Databricks App config
    requirements.txt           # App dependencies
  data/
    canonical_columns.csv      # Seed canonical concept definitions
    column_mappings.csv        # Seed approved examples (Type 2 SCD)
    column_rules.csv           # Seed rules
    salesforce_customers.csv   # Demo source data
    shopify_orders.csv         # Demo source data
  notebooks/
    01_setup_data_and_vector_search.py   # Bootstrap tables + index
    02_run_column_mapping_agentic.py     # Batch mapping job
  src/column_mapping/
    agentic_mapping.py         # Vector search + LLM proposal logic
    config.py                  # Config loading and resolution
    demo_data.py               # Demo table creation from CSVs
    rules.py                   # Rule loading and prompt formatting
    vector_search.py           # Databricks Vector Search helpers
    workspace_paths.py         # Workspace path resolution
  config.yaml                 # Default configuration
  databricks.yml               # Databricks Asset Bundle definition
  pyproject.toml               # Python project config
```

---

## Configuration

All configuration lives in `config.yaml` with notebook widget overrides for job parameterization.

| Setting | Purpose |
|---|---|
| `databricks.catalog` | Unity Catalog catalog for all tables |
| `databricks.schema` | Schema within the catalog |
| `databricks.table_prefix` | Prefix for discovering source tables |
| `tables.rules_table_name` | Name of the rules table |
| `tables.mappings_table_name` | Name of the mappings table |
| `tables.canonical_columns_table_name` | Name of the canonical columns table |
| `vector_search.endpoint_name` | Databricks VS endpoint (batch jobs) |
| `vector_search.index_name` | VS index name (batch jobs) |
| `llm.endpoint` | Model serving endpoint for proposals |
| `mapping.top_k` | Number of similar examples to retrieve |
