"""
Column Mapping Review App

Multi-tab Streamlit app that guides data stewards through the column
standardization workflow:
  1. Source Systems  -- discover what's there
  2. Review Mappings -- approve / reject / edit LLM proposals
  3. Canonical Map   -- cross-system pivot of canonical concepts
  4. Rules           -- manage naming rules
"""

import os
import sys
import traceback
import uuid
from datetime import datetime, timezone

import numpy as np


def _log(msg):
    print(f"[app] {msg}", file=sys.stderr, flush=True)


_log("app.py: module load starting")

import pandas as pd
import streamlit as st
from databricks.sdk import WorkspaceClient

try:
    import faiss
    from sklearn.feature_extraction.text import TfidfVectorizer

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    _log("faiss or scikit-learn not available; similarity search disabled")

_log(f"streamlit {st.__version__} on Python {sys.version_info[:2]}")

st.set_page_config(page_title="Column Mapping Review", layout="wide")

_DATABRICKS_CSS = """
<style>
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'DM Sans', sans-serif;
        color: #FF3621 !important;
    }
    [data-testid="stSidebarContent"] h1,
    [data-testid="stSidebarContent"] h2,
    [data-testid="stSidebarContent"] h3 { color: #FF3621 !important; }
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background-color: #FF3621; border-color: #FF3621;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background-color: #E02E1B; border-color: #E02E1B;
    }
    [data-testid="stMetricValue"] { color: #1B3139; }
    hr { border-color: #FF3621 !important; opacity: 0.3; }
</style>
"""
st.markdown(_DATABRICKS_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Configuration -- env vars > config.yaml > defaults
# ---------------------------------------------------------------------------

METADATA_TABLES = {
    "standardization_mappings", "standardization_rules",
    "canonical_columns", "column_approvals",
}


def _load_config_yaml():
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "config.yaml"),
        "config.yaml",
    ]
    for path in candidates:
        path = os.path.abspath(path)
        if os.path.isfile(path):
            _log(f"loading config from {path}")
            with open(path) as f:
                return yaml.safe_load(f) or {}
    return {}


def _resolve_config():
    cfg = _load_config_yaml()
    db = cfg.get("databricks", {})
    tables = cfg.get("tables", {})

    catalog = db.get("catalog", "shm")
    schema = db.get("schema", "columnmapping")

    mappings = os.getenv("MAPPINGS_TABLE") or (
        f"{catalog}.{schema}.{tables.get('mappings_table_name', 'standardization_mappings')}"
    )
    canonical = os.getenv("CANONICAL_TABLE") or (
        f"{catalog}.{schema}.{tables.get('canonical_columns_table_name', 'canonical_columns')}"
    )
    rules = os.getenv("RULES_TABLE") or (
        f"{catalog}.{schema}.{tables.get('rules_table_name', 'standardization_rules')}"
    )
    warehouse = os.getenv("DATABRICKS_WAREHOUSE_ID") or db.get("warehouse_id", "")

    return catalog, schema, mappings, canonical, rules, warehouse


CATALOG, SCHEMA, MAPPINGS_TABLE, CANONICAL_TABLE, RULES_TABLE, WAREHOUSE_ID = (
    _resolve_config()
)

_log(f"config: {CATALOG}.{SCHEMA}")
_log(f"config: WAREHOUSE_ID={'(set)' if WAREHOUSE_ID else '(auto-discover)'}")

DISPLAY_COLUMNS = [
    "mapping_id", "version", "source_system", "platform_header",
    "standardized_header", "domain", "data_type", "confidence_score",
    "transformation_notes", "approval_status",
]

# ---------------------------------------------------------------------------
# Databricks helpers
# ---------------------------------------------------------------------------


@st.cache_resource
def get_workspace_client():
    _log("creating WorkspaceClient()")
    wc = WorkspaceClient()
    _log("WorkspaceClient created ok")
    return wc


def get_current_user():
    try:
        headers = st.context.headers
        for key in (
            "X-Forwarded-Email",
            "X-Forwarded-Preferred-Username",
            "X-Forwarded-User",
        ):
            val = headers.get(key)
            if val:
                return val
    except Exception:
        pass
    try:
        me = get_workspace_client().current_user.me()
        return me.user_name or me.display_name or "unknown"
    except Exception:
        return "unknown"


def _esc(val):
    if val is None:
        return "NULL"
    return "'" + str(val).replace("'", "''") + "'"


def _get_warehouse_id():
    global WAREHOUSE_ID
    if WAREHOUSE_ID:
        return WAREHOUSE_ID

    _log("auto-discovering SQL warehouse...")
    ws = get_workspace_client()
    for wh in ws.warehouses.list():
        if wh.state and wh.state.value in ("RUNNING", "STARTING"):
            WAREHOUSE_ID = wh.id
            _log(f"using warehouse: {wh.name} ({wh.id})")
            return WAREHOUSE_ID
    for wh in ws.warehouses.list():
        if wh.id:
            WAREHOUSE_ID = wh.id
            _log(f"using warehouse (stopped): {wh.name} ({wh.id})")
            return WAREHOUSE_ID
    raise RuntimeError(
        "No SQL warehouse found. Set warehouse_id in config.yaml "
        "or DATABRICKS_WAREHOUSE_ID env var."
    )


def execute_query(query):
    wid = _get_warehouse_id()
    ws = get_workspace_client()
    resp = ws.statement_execution.execute_statement(
        warehouse_id=wid, statement=query, wait_timeout="30s",
    )
    if resp.status and resp.status.error:
        raise RuntimeError(f"SQL error: {resp.status.error.message}")
    if resp.result and resp.result.data_array:
        columns = [c.name for c in resp.manifest.schema.columns]
        return [dict(zip(columns, row)) for row in resp.result.data_array]
    return []


def execute_statement(query):
    wid = _get_warehouse_id()
    ws = get_workspace_client()
    resp = ws.statement_execution.execute_statement(
        warehouse_id=wid, statement=query, wait_timeout="30s",
    )
    if resp.status and resp.status.error:
        raise RuntimeError(f"SQL error: {resp.status.error.message}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_current_mappings():
    _log(f"load_current_mappings: querying {MAPPINGS_TABLE}")
    rows = execute_query(
        f"""
        SELECT
            mapping_id,
            CAST(version AS INT) AS version,
            source_system,
            platform_header,
            standardized_header,
            domain,
            data_type,
            CAST(confidence_score AS INT) AS confidence_score,
            COALESCE(transformation_notes, '') AS transformation_notes,
            approval_status
        FROM {MAPPINGS_TABLE}
        WHERE valid_to IS NULL
        ORDER BY confidence_score DESC, source_system, platform_header
        """
    )
    df = pd.DataFrame(rows, columns=DISPLAY_COLUMNS)
    df["confidence_score"] = (
        pd.to_numeric(df["confidence_score"], errors="coerce").fillna(0).astype(int)
    )
    df["version"] = (
        pd.to_numeric(df["version"], errors="coerce").fillna(1).astype(int)
    )
    _log(f"load_current_mappings: loaded {len(df)} current rows")
    return df


def load_source_tables():
    _log("load_source_tables: querying information_schema")
    rows = execute_query(
        f"""
        SELECT
            t.table_name,
            COUNT(c.column_name) AS column_count
        FROM {CATALOG}.information_schema.tables t
        JOIN {CATALOG}.information_schema.columns c
            ON t.table_catalog = c.table_catalog
            AND t.table_schema = c.table_schema
            AND t.table_name = c.table_name
        WHERE t.table_catalog = '{CATALOG}'
          AND t.table_schema = '{SCHEMA}'
          AND t.table_type IN ('MANAGED', 'EXTERNAL')
        GROUP BY t.table_name
        ORDER BY t.table_name
        """
    )
    return [
        r for r in rows
        if r["table_name"] not in METADATA_TABLES
        and not r["table_name"].endswith("_index")
    ]


def load_canonical_columns():
    _log(f"load_canonical_columns: querying {CANONICAL_TABLE}")
    rows = execute_query(
        f"""
        SELECT canonical_name, business_definition, domain, expected_data_type
        FROM {CANONICAL_TABLE}
        ORDER BY canonical_name
        """
    )
    return rows


def load_rules():
    _log(f"load_rules: querying {RULES_TABLE}")
    rows = execute_query(
        f"""
        SELECT rule_id, rule_type, rule_key, rule_value,
               rule_description, examples, is_active
        FROM {RULES_TABLE}
        WHERE is_active = true
        ORDER BY rule_type, rule_key
        """
    )
    return rows


def get_mappings_df():
    if "mappings_df" not in st.session_state:
        st.session_state["mappings_df"] = load_current_mappings()
    return st.session_state["mappings_df"]


def get_source_tables():
    if "source_tables" not in st.session_state:
        st.session_state["source_tables"] = load_source_tables()
    return st.session_state["source_tables"]


def get_canonical_columns():
    if "canonical_columns" not in st.session_state:
        st.session_state["canonical_columns"] = load_canonical_columns()
    return st.session_state["canonical_columns"]


def get_rules():
    if "rules" not in st.session_state:
        st.session_state["rules"] = load_rules()
    return st.session_state["rules"]


def apply_local_edits(df):
    edits = st.session_state.get("pending_edits", {})
    if not edits:
        return df
    df = df.copy()
    for mapping_id, edit in edits.items():
        mask = df["mapping_id"] == mapping_id
        if mask.any():
            for field in ("approval_status", "standardized_header", "domain", "data_type"):
                if field in edit:
                    df.loc[mask, field] = edit[field]
    return df


# ---------------------------------------------------------------------------
# FAISS in-memory vector search
# ---------------------------------------------------------------------------


def build_faiss_index(approved_df):
    if not FAISS_AVAILABLE or approved_df.empty:
        return None, None, None

    texts = approved_df["platform_header"].tolist()
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    tfidf_matrix = vectorizer.fit_transform(texts)

    vectors = np.ascontiguousarray(tfidf_matrix.toarray().astype("float32"))
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    _log(f"built FAISS index: {len(texts)} vectors, {vectors.shape[1]} dimensions")
    return index, vectorizer, approved_df.reset_index(drop=True)


def get_faiss_index():
    if "faiss_index" not in st.session_state:
        df = get_mappings_df()
        approved = df[df["approval_status"] == "approved"]
        st.session_state["faiss_index"] = build_faiss_index(approved)
    return st.session_state["faiss_index"]


def search_similar_mappings(query, top_k=5):
    index, vectorizer, approved_df = get_faiss_index()
    if index is None or approved_df is None or approved_df.empty:
        return pd.DataFrame()

    query_vec = vectorizer.transform([query]).toarray().astype("float32")
    faiss.normalize_L2(query_vec)

    k = min(top_k, len(approved_df))
    scores, indices = index.search(query_vec, k)

    results = approved_df.iloc[indices[0]].copy()
    results["similarity"] = (scores[0] * 100).astype(int)
    return results[results["similarity"] > 0].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Session-state edit tracking
# ---------------------------------------------------------------------------


def init_session_state():
    if "pending_edits" not in st.session_state:
        st.session_state["pending_edits"] = {}


def stage_status_change(mapping_ids, new_status, user):
    now = datetime.now(timezone.utc).isoformat()
    for mid in mapping_ids:
        existing = st.session_state["pending_edits"].get(mid, {})
        existing.update({
            "mapping_id": mid,
            "approval_status": new_status,
            "changed_by": user,
            "changed_at": now,
            "change_reason": f"{new_status.replace('_', ' ').title()} by steward",
        })
        st.session_state["pending_edits"][mid] = existing
    return len(mapping_ids)


def stage_field_edit(mapping_id, field, value, user):
    now = datetime.now(timezone.utc).isoformat()
    existing = st.session_state["pending_edits"].get(mapping_id, {})
    old_value = existing.get(f"_old_{field}")
    existing.update({
        "mapping_id": mapping_id,
        field: value,
        "changed_by": user,
        "changed_at": now,
    })
    reason_parts = existing.get("change_reason", "").split("; ")
    reason_parts = [r for r in reason_parts if not r.startswith(f"Edited {field}")]
    if old_value is None:
        df = get_mappings_df()
        row = df[df["mapping_id"] == mapping_id]
        if not row.empty:
            old_value = row.iloc[0][field]
            existing[f"_old_{field}"] = old_value
    reason_parts.append(f"Edited {field}: {old_value} -> {value}")
    existing["change_reason"] = "; ".join(r for r in reason_parts if r)
    st.session_state["pending_edits"][mapping_id] = existing


def discard_edits():
    st.session_state["pending_edits"] = {}


# ---------------------------------------------------------------------------
# Canonical column auto-creation
# ---------------------------------------------------------------------------


def ensure_canonical_column(standardized_header, domain, data_type, created_by):
    existing = get_canonical_columns()
    known_names = {c["canonical_name"] for c in existing}
    if standardized_header in known_names:
        return

    execute_statement(
        f"""
        INSERT INTO {CANONICAL_TABLE}
            (canonical_name, business_definition, domain,
             expected_data_type, created_by, created_at)
        VALUES
            ({_esc(standardized_header)},
             {_esc('Auto-created from mapping approval')},
             {_esc(domain)}, {_esc(data_type)},
             {_esc(created_by)}, current_timestamp())
        """
    )
    _log(f"auto-created canonical column: {standardized_header}")
    st.session_state.pop("canonical_columns", None)


# ---------------------------------------------------------------------------
# Commit: Type 2 SCD version insert
# ---------------------------------------------------------------------------


def commit_edits():
    edits = st.session_state.get("pending_edits", {})
    if not edits:
        return 0

    df = get_mappings_df()

    for edit in edits.values():
        mid = edit["mapping_id"]
        row = df[df["mapping_id"] == mid]
        if row.empty:
            continue

        current = row.iloc[0]
        new_version = int(current["version"]) + 1

        new_status = edit.get("approval_status", current["approval_status"])
        new_std_header = edit.get("standardized_header", current["standardized_header"])
        new_domain = edit.get("domain", current["domain"])
        new_data_type = edit.get("data_type", current["data_type"])
        changed_by = edit.get("changed_by", "unknown")
        change_reason = edit.get("change_reason", "Updated by steward")

        execute_statement(
            f"""
            UPDATE {MAPPINGS_TABLE}
            SET valid_to = current_timestamp()
            WHERE mapping_id = {_esc(mid)} AND valid_to IS NULL
            """
        )

        execute_statement(
            f"""
            INSERT INTO {MAPPINGS_TABLE}
                (mapping_id, version, source_system, platform_header,
                 standardized_header, domain, data_type,
                 transformation_notes, confidence_score, approval_status,
                 valid_from, valid_to, changed_by, change_reason)
            VALUES
                ({_esc(mid)}, {new_version}, {_esc(current['source_system'])},
                 {_esc(current['platform_header'])},
                 {_esc(new_std_header)}, {_esc(new_domain)}, {_esc(new_data_type)},
                 {_esc(current['transformation_notes'])},
                 {int(current['confidence_score'])}, {_esc(new_status)},
                 current_timestamp(), NULL,
                 {_esc(changed_by)}, {_esc(change_reason)})
            """
        )

        if new_status == "approved":
            ensure_canonical_column(new_std_header, new_domain, new_data_type, changed_by)

    count = len(edits)
    discard_edits()
    _invalidate_caches()
    return count


def _invalidate_caches():
    for key in ("mappings_df", "canonical_columns", "rules", "faiss_index", "source_tables"):
        st.session_state.pop(key, None)
    st.session_state["mappings_df"] = load_current_mappings()


# ---------------------------------------------------------------------------
# Add rule
# ---------------------------------------------------------------------------


def insert_rule(rule_type, rule_key, rule_value, rule_description, examples):
    rule_id = f"{rule_type[:3].upper()}-{uuid.uuid4().hex[:6].upper()}"
    execute_statement(
        f"""
        INSERT INTO {RULES_TABLE}
            (rule_id, rule_type, rule_key, rule_value,
             rule_description, examples, is_active)
        VALUES
            ({_esc(rule_id)}, {_esc(rule_type)}, {_esc(rule_key)},
             {_esc(rule_value)}, {_esc(rule_description)},
             {_esc(examples)}, true)
        """
    )
    st.session_state.pop("rules", None)
    return rule_id


# ---------------------------------------------------------------------------
# Mapping history
# ---------------------------------------------------------------------------


def load_mapping_history(mapping_id):
    rows = execute_query(
        f"""
        SELECT mapping_id, version, approval_status,
               standardized_header, domain, data_type,
               valid_from, valid_to, changed_by, change_reason
        FROM {MAPPINGS_TABLE}
        WHERE mapping_id = {_esc(mapping_id)}
        ORDER BY version
        """
    )
    return pd.DataFrame(rows)


# =========================================================================
# STREAMLIT UI
# =========================================================================

init_session_state()

try:
    current_user = get_current_user()
except Exception:
    current_user = "unknown"

pending_edits = st.session_state["pending_edits"]

# -- sidebar ---------------------------------------------------------------
st.sidebar.markdown(f"**Signed in as:** `{current_user}`")
st.sidebar.caption(f"`{CATALOG}`.`{SCHEMA}`")
st.sidebar.divider()

if pending_edits:
    st.sidebar.warning(f"{len(pending_edits)} uncommitted edit(s)")
    sb_c1, sb_c2 = st.sidebar.columns(2)
    with sb_c1:
        if st.button("Commit", type="primary", key="sb_commit"):
            n = commit_edits()
            st.toast(f"Committed {n} decision(s).")
            st.rerun()
    with sb_c2:
        if st.button("Discard", key="sb_discard"):
            discard_edits()
            st.toast("Edits discarded.")
            st.rerun()
    st.sidebar.divider()

if st.sidebar.button("Reload data"):
    _invalidate_caches()
    st.rerun()

# -- load data --------------------------------------------------------------
try:
    raw_df = get_mappings_df()
    source_tables = get_source_tables()
    canonical = get_canonical_columns()
    rules_data = get_rules()
except Exception as exc:
    _tb = traceback.format_exc()
    _log(f"data load failed: {exc}\n{_tb}")
    st.error(f"Failed to load data: {exc}")
    with st.expander("Debug info"):
        st.code(_tb)
    st.stop()

display_df = apply_local_edits(raw_df)

canonical_domains = sorted(set(
    c.get("domain", "") for c in canonical if c.get("domain")
))
canonical_types = sorted(set(
    c.get("expected_data_type", "") for c in canonical if c.get("expected_data_type")
))

# -- tabs -------------------------------------------------------------------
tab_sources, tab_review, tab_canonical, tab_rules = st.tabs([
    "Source Systems", "Review Mappings", "Canonical Map", "Rules",
])

# =========================================================================
# TAB 1: SOURCE SYSTEMS
# =========================================================================
with tab_sources:
    st.header("Source Systems")
    st.caption("Tables discovered in the catalog. Each source system's columns "
               "are mapped to canonical concepts by the standardization pipeline.")

    source_names = set(r["table_name"] for r in source_tables)
    mapped_sources = display_df["source_system"].unique()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Source tables", len(source_tables))
    m2.metric("Total source columns", sum(int(r["column_count"]) for r in source_tables))
    m3.metric("Mapped columns", len(display_df))
    approved_count = int((display_df["approval_status"] == "approved").sum())
    m4.metric("Approved mappings", approved_count)

    st.divider()

    for tbl in source_tables:
        tbl_name = tbl["table_name"]
        col_count = int(tbl["column_count"])

        tbl_mappings = display_df[display_df["source_system"] == tbl_name]
        mapped_count = len(tbl_mappings)
        approved = int((tbl_mappings["approval_status"] == "approved").sum())
        pending = int((tbl_mappings["approval_status"] == "pending_review").sum())
        rejected = int((tbl_mappings["approval_status"] == "rejected").sum())

        coverage = mapped_count / col_count if col_count > 0 else 0

        with st.expander(
            f"**{tbl_name}** -- {col_count} columns, "
            f"{mapped_count} mapped ({coverage:.0%} coverage)",
            expanded=False,
        ):
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            sc1.metric("Columns", col_count)
            sc2.metric("Mapped", mapped_count)
            sc3.metric("Approved", approved)
            sc4.metric("Pending", pending)
            sc5.metric("Rejected", rejected)

            if not tbl_mappings.empty:
                st.dataframe(
                    tbl_mappings[["platform_header", "standardized_header",
                                  "domain", "data_type", "confidence_score",
                                  "approval_status"]].reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No mappings yet. Run the agentic mapping job to generate proposals.")


# =========================================================================
# TAB 2: REVIEW MAPPINGS
# =========================================================================
with tab_review:
    st.header("Review Mappings")
    st.caption(
        "Filter, select, and approve or reject LLM-proposed column mappings. "
        "Edits are staged locally until you commit."
    )

    # -- filters in columns instead of sidebar for this tab
    fc1, fc2, fc3, fc4 = st.columns(4)

    all_statuses = ["All"] + sorted(display_df["approval_status"].dropna().unique().tolist())
    all_sources = ["All"] + sorted(display_df["source_system"].dropna().unique().tolist())
    all_domains = ["All"] + sorted(display_df["domain"].dropna().unique().tolist())

    status_filter = fc1.selectbox("Status", all_statuses, index=0, key="r_status")
    source_filter = fc2.selectbox("Source system", all_sources, index=0, key="r_source")
    domain_filter = fc3.selectbox("Domain", all_domains, index=0, key="r_domain")
    min_confidence = fc4.slider("Min confidence", 0, 100, 0, key="r_conf")

    filtered_df = display_df.copy()
    if status_filter != "All":
        filtered_df = filtered_df[filtered_df["approval_status"] == status_filter]
    if source_filter != "All":
        filtered_df = filtered_df[filtered_df["source_system"] == source_filter]
    if domain_filter != "All":
        filtered_df = filtered_df[filtered_df["domain"] == domain_filter]
    filtered_df = filtered_df[filtered_df["confidence_score"] >= min_confidence]

    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Showing", len(filtered_df))
    if len(filtered_df):
        rc2.metric("Avg confidence", f"{filtered_df['confidence_score'].mean():.0f}")
        rc3.metric("Pending review", int((filtered_df["approval_status"] == "pending_review").sum()))
    rc4.metric("Uncommitted edits", len(pending_edits))

    if filtered_df.empty:
        st.info("No mappings match the current filters.")
    else:
        styled_df = filtered_df.reset_index(drop=True)

        event = st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
        )

        selected_indices = event.selection.rows if event.selection else []
        selected_ids = [styled_df.iloc[i]["mapping_id"] for i in selected_indices]

        # -- action buttons
        if not selected_ids:
            st.info("Select one or more rows above, then choose an action.")
        else:
            st.write(f"**{len(selected_ids)}** mapping(s) selected.")

        act_c1, act_c2, act_c3, act_c4, _ = st.columns([1, 1, 1, 1, 3])

        with act_c1:
            if st.button("Approve", type="primary", disabled=not selected_ids, key="rv_approve"):
                n = stage_status_change(selected_ids, "approved", current_user)
                st.toast(f"Staged {n} approval(s).")
                st.rerun()
        with act_c2:
            if st.button("Reject", disabled=not selected_ids, key="rv_reject"):
                n = stage_status_change(selected_ids, "rejected", current_user)
                st.toast(f"Staged {n} rejection(s).")
                st.rerun()
        with act_c3:
            if st.button("Commit all", type="primary", disabled=not pending_edits, key="rv_commit"):
                n = commit_edits()
                st.toast(f"Committed {n} decision(s).")
                st.rerun()
        with act_c4:
            if st.button("Discard all", disabled=not pending_edits, key="rv_discard"):
                discard_edits()
                st.toast("All edits discarded.")
                st.rerun()

        # -- edit single mapping
        if len(selected_ids) == 1:
            st.divider()
            st.subheader("Edit mapping")

            sel_row = styled_df[styled_df["mapping_id"] == selected_ids[0]].iloc[0]

            edit_c1, edit_c2, edit_c3 = st.columns(3)

            with edit_c1:
                new_std = st.text_input(
                    "Standardized header",
                    value=sel_row["standardized_header"],
                    key="edit_std_header",
                )
                if new_std != sel_row["standardized_header"]:
                    stage_field_edit(selected_ids[0], "standardized_header", new_std, current_user)

            with edit_c2:
                domain_options = canonical_domains if canonical_domains else [sel_row["domain"]]
                if sel_row["domain"] not in domain_options:
                    domain_options = [sel_row["domain"]] + domain_options
                domain_idx = (
                    domain_options.index(sel_row["domain"])
                    if sel_row["domain"] in domain_options else 0
                )
                new_domain = st.selectbox(
                    "Domain", domain_options, index=domain_idx, key="edit_domain"
                )
                if new_domain != sel_row["domain"]:
                    stage_field_edit(selected_ids[0], "domain", new_domain, current_user)

            with edit_c3:
                type_options = canonical_types if canonical_types else [sel_row["data_type"]]
                if sel_row["data_type"] not in type_options:
                    type_options = [sel_row["data_type"]] + type_options
                type_idx = (
                    type_options.index(sel_row["data_type"])
                    if sel_row["data_type"] in type_options else 0
                )
                new_type = st.selectbox(
                    "Data type", type_options, index=type_idx, key="edit_data_type"
                )
                if new_type != sel_row["data_type"]:
                    stage_field_edit(selected_ids[0], "data_type", new_type, current_user)

            detail_c1, detail_c2 = st.columns(2)

            with detail_c1:
                if FAISS_AVAILABLE:
                    with st.expander("Similar approved mappings", expanded=True):
                        similar = search_similar_mappings(sel_row["platform_header"], top_k=5)
                        if not similar.empty:
                            st.dataframe(
                                similar[["platform_header", "standardized_header", "domain",
                                         "data_type", "source_system", "similarity"]],
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.info("No similar approved mappings found.")

            with detail_c2:
                with st.expander("Version history", expanded=True):
                    hist_df = load_mapping_history(selected_ids[0])
                    if not hist_df.empty:
                        st.dataframe(hist_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No history found.")

    # -- pending edits summary
    if pending_edits:
        st.divider()
        st.subheader("Pending edits (uncommitted)")
        display_edits = [
            {k: v for k, v in edit.items() if not k.startswith("_")}
            for edit in pending_edits.values()
        ]
        st.dataframe(pd.DataFrame(display_edits), use_container_width=True, hide_index=True)


# =========================================================================
# TAB 3: CANONICAL MAP
# =========================================================================
with tab_canonical:
    st.header("Canonical Map")
    st.caption(
        "Cross-system view: for every canonical concept, which source systems "
        "have a column that maps to it and what do they call it?"
    )

    approved_df = display_df[display_df["approval_status"] == "approved"]

    if approved_df.empty:
        st.info("No approved mappings yet. Approve some mappings in the Review tab first.")
    else:
        # Build the pivot: canonical_name x source_system -> platform_header
        canon_df = pd.DataFrame(canonical)
        pivot_data = approved_df[["source_system", "platform_header", "standardized_header"]].copy()
        pivot_data = pivot_data.rename(columns={"standardized_header": "canonical_name"})

        all_systems = sorted(pivot_data["source_system"].unique())

        # Domain filter for canonical map
        cm_domain_filter = st.selectbox(
            "Filter by domain", ["All"] + canonical_domains, index=0, key="cm_domain"
        )

        canon_names = canon_df["canonical_name"].tolist()
        if cm_domain_filter != "All":
            canon_names = canon_df[canon_df["domain"] == cm_domain_filter]["canonical_name"].tolist()

        pivot_rows = []
        for cn in canon_names:
            cn_info = canon_df[canon_df["canonical_name"] == cn]
            row = {
                "canonical_name": cn,
                "domain": cn_info.iloc[0]["domain"] if not cn_info.empty else "",
                "data_type": cn_info.iloc[0]["expected_data_type"] if not cn_info.empty else "",
            }
            cn_mappings = pivot_data[pivot_data["canonical_name"] == cn]
            for sys_name in all_systems:
                sys_cols = cn_mappings[cn_mappings["source_system"] == sys_name]["platform_header"]
                row[sys_name] = ", ".join(sys_cols.tolist()) if not sys_cols.empty else ""
            system_count = sum(1 for s in all_systems if row.get(s, ""))
            row["systems"] = system_count
            pivot_rows.append(row)

        # Also add canonical names that exist only in mappings (not in canonical table)
        mapped_canonicals = set(pivot_data["canonical_name"].unique())
        known_canonicals = set(canon_df["canonical_name"].tolist())
        for cn in sorted(mapped_canonicals - known_canonicals):
            row = {"canonical_name": cn, "domain": "", "data_type": "", "systems": 0}
            cn_mappings = pivot_data[pivot_data["canonical_name"] == cn]
            for sys_name in all_systems:
                sys_cols = cn_mappings[cn_mappings["source_system"] == sys_name]["platform_header"]
                row[sys_name] = ", ".join(sys_cols.tolist()) if not sys_cols.empty else ""
            row["systems"] = sum(1 for s in all_systems if row.get(s, ""))
            if cm_domain_filter == "All":
                pivot_rows.append(row)

        if pivot_rows:
            pivot_df = pd.DataFrame(pivot_rows)
            ordered_cols = ["canonical_name", "domain", "data_type", "systems"] + all_systems
            pivot_df = pivot_df[[c for c in ordered_cols if c in pivot_df.columns]]
            pivot_df = pivot_df.sort_values(["systems", "canonical_name"], ascending=[False, True])

            cm1, cm2, cm3 = st.columns(3)
            cm1.metric("Canonical concepts", len(pivot_df))
            cm2.metric("Source systems", len(all_systems))
            multi_system = int((pivot_df["systems"] >= 2).sum())
            cm3.metric("Cross-system concepts", multi_system)

            st.dataframe(
                pivot_df.reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
                height=600,
            )
        else:
            st.info("No canonical concepts match the current filter.")


# =========================================================================
# TAB 4: RULES
# =========================================================================
with tab_rules:
    st.header("Rules")
    st.caption(
        "Naming rules constrain LLM proposals. Abbreviation rules, naming conventions, "
        "domain classifications, and data type mappings."
    )

    if rules_data:
        rules_df = pd.DataFrame(rules_data)

        rule_type_filter = st.selectbox(
            "Filter by type",
            ["All"] + sorted(rules_df["rule_type"].unique().tolist()),
            key="rules_type_filter",
        )
        if rule_type_filter != "All":
            rules_df = rules_df[rules_df["rule_type"] == rule_type_filter]

        rm1, rm2, rm3, rm4 = st.columns(4)
        rm1.metric("Total rules", len(rules_df))
        for i, rt in enumerate(["abbreviation", "naming_convention", "domain", "data_type"]):
            count = int((rules_df["rule_type"] == rt).sum()) if not rules_df.empty else 0
            [rm2, rm3, rm4, rm1][i].metric(rt.replace("_", " ").title(), count)

        st.dataframe(
            rules_df[["rule_id", "rule_type", "rule_key", "rule_value",
                       "rule_description", "examples"]].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No active rules found.")

    st.divider()
    st.subheader("Add a rule")

    rule_types = ["abbreviation", "naming_convention", "domain", "data_type"]

    ar_c1, ar_c2 = st.columns(2)
    with ar_c1:
        new_rule_type = st.selectbox("Rule type", rule_types, key="new_rule_type")
        new_rule_key = st.text_input("Key", key="new_rule_key",
                                     placeholder="e.g. acct, mgr, amt")
    with ar_c2:
        new_rule_value = st.text_input("Value", key="new_rule_value",
                                       placeholder="e.g. account, mgr, amt")
        new_rule_desc = st.text_input("Description", key="new_rule_desc",
                                      placeholder="e.g. Expand acct to account")

    new_rule_examples = st.text_input("Examples", key="new_rule_examples",
                                      placeholder="e.g. Acct-ID -> account_id")

    can_add = all([new_rule_key, new_rule_value, new_rule_desc])
    if st.button("Add rule", type="primary", disabled=not can_add, key="btn_add_rule"):
        rid = insert_rule(
            new_rule_type, new_rule_key, new_rule_value,
            new_rule_desc, new_rule_examples or ""
        )
        st.toast(f"Added rule {rid}.")
        st.rerun()

_log("render complete")
