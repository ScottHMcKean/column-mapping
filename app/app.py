"""
Column Mapping Review App

Loads the full mappings table into memory once, filters locally,
tracks edits in session state, and commits decisions as append-only
rows to a separate ``column_approvals`` table.
"""

import os
import sys
import traceback
from datetime import datetime, timezone


def _log(msg):
    print(f"[app] {msg}", file=sys.stderr, flush=True)


_log("app.py: module load starting")

import pandas as pd
import streamlit as st
from databricks.sdk import WorkspaceClient

_log(f"streamlit {st.__version__} on Python {sys.version_info[:2]}")

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Column Mapping Review", layout="wide")

_DATABRICKS_CSS = """
<style>
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'DM Sans', sans-serif;
        color: #FF3621 !important;
    }
    [data-testid="stSidebarContent"] h1,
    [data-testid="stSidebarContent"] h2,
    [data-testid="stSidebarContent"] h3 {
        color: #FF3621 !important;
    }
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background-color: #FF3621;
        border-color: #FF3621;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background-color: #E02E1B;
        border-color: #E02E1B;
    }
    [data-testid="stMetricValue"] {
        color: #1B3139;
    }
    hr {
        border-color: #FF3621 !important;
        opacity: 0.3;
    }
</style>
"""
st.markdown(_DATABRICKS_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAPPINGS_TABLE = os.getenv(
    "MAPPINGS_TABLE", "shm.columnmapping.standardization_mappings"
)
APPROVALS_TABLE = os.getenv("APPROVALS_TABLE", "shm.columnmapping.column_approvals")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "")

_log(f"config: MAPPINGS_TABLE={MAPPINGS_TABLE}")
_log(f"config: WAREHOUSE_ID={'(set)' if WAREHOUSE_ID else '(NOT SET)'}")

DISPLAY_COLUMNS = [
    "mapping_id",
    "source_system",
    "platform_header",
    "standardized_header",
    "domain",
    "data_type",
    "confidence_score",
    "transformation_notes",
    "approval_status",
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
    """Resolve the authenticated user email via OBO headers or SDK."""
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


def execute_query(query):
    """Run a read-only SQL statement via the Statement Execution API."""
    if not WAREHOUSE_ID:
        raise RuntimeError("DATABRICKS_WAREHOUSE_ID environment variable is not set.")
    ws = get_workspace_client()
    resp = ws.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=query,
        wait_timeout="30s",
    )
    if resp.status and resp.status.error:
        raise RuntimeError(f"SQL error: {resp.status.error.message}")
    if resp.result and resp.result.data_array:
        columns = [c.name for c in resp.manifest.schema.columns]
        return [dict(zip(columns, row)) for row in resp.result.data_array]
    return []


def execute_statement(query):
    """Run a DML / DDL statement."""
    if not WAREHOUSE_ID:
        raise RuntimeError("DATABRICKS_WAREHOUSE_ID environment variable is not set.")
    ws = get_workspace_client()
    resp = ws.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=query,
        wait_timeout="30s",
    )
    if resp.status and resp.status.error:
        raise RuntimeError(f"SQL error: {resp.status.error.message}")


# ---------------------------------------------------------------------------
# One-time data load (cached in session state)
# ---------------------------------------------------------------------------


def load_full_table():
    """Fetch the entire mappings table and cache it in session state."""
    _log(f"load_full_table: querying {MAPPINGS_TABLE}")
    rows = execute_query(
        f"""
        SELECT
            mapping_id,
            source_system,
            platform_header,
            standardized_header,
            domain,
            data_type,
            CAST(confidence_score AS INT) AS confidence_score,
            COALESCE(transformation_notes, '') AS transformation_notes,
            approval_status
        FROM {MAPPINGS_TABLE}
        ORDER BY confidence_score DESC, source_system, platform_header
        """
    )
    df = pd.DataFrame(rows, columns=DISPLAY_COLUMNS)
    df["confidence_score"] = (
        pd.to_numeric(df["confidence_score"], errors="coerce").fillna(0).astype(int)
    )
    _log(f"load_full_table: loaded {len(df)} rows")
    return df


def get_mappings_df():
    """Return the in-memory mappings DataFrame, loading on first access."""
    if "mappings_df" not in st.session_state:
        st.session_state["mappings_df"] = load_full_table()
    return st.session_state["mappings_df"]


def apply_local_edits(df):
    """Overlay any pending session edits onto the cached DataFrame."""
    edits = st.session_state.get("pending_edits", {})
    if not edits:
        return df
    df = df.copy()
    for mapping_id, edit in edits.items():
        mask = df["mapping_id"] == mapping_id
        if mask.any():
            df.loc[mask, "approval_status"] = edit["new_status"]
    return df


# ---------------------------------------------------------------------------
# Session-state edit tracking
# ---------------------------------------------------------------------------


def init_session_state():
    if "pending_edits" not in st.session_state:
        st.session_state["pending_edits"] = {}


def stage_edits(mapping_ids, new_status, user):
    now = datetime.now(timezone.utc).isoformat()
    for mid in mapping_ids:
        st.session_state["pending_edits"][mid] = {
            "mapping_id": mid,
            "new_status": new_status,
            "decided_by": user,
            "decided_at": now,
        }
    return len(mapping_ids)


def discard_edits():
    st.session_state["pending_edits"] = {}


# ---------------------------------------------------------------------------
# Commit: append-only write to column_approvals
# ---------------------------------------------------------------------------


def ensure_approvals_table():
    execute_statement(
        f"""
        CREATE TABLE IF NOT EXISTS {APPROVALS_TABLE} (
            mapping_id       STRING    NOT NULL,
            new_status       STRING    NOT NULL,
            decided_by       STRING    NOT NULL,
            decided_at       TIMESTAMP NOT NULL,
            committed_at     TIMESTAMP NOT NULL
        )
        """
    )


def commit_edits():
    edits = st.session_state.get("pending_edits", {})
    if not edits:
        return 0

    ensure_approvals_table()

    value_rows = []
    for edit in edits.values():
        mid = edit["mapping_id"].replace("'", "''")
        status = edit["new_status"].replace("'", "''")
        user = edit["decided_by"].replace("'", "''")
        ts = edit["decided_at"].replace("'", "''")
        value_rows.append(
            f"('{mid}', '{status}', '{user}', '{ts}', current_timestamp())"
        )

    values_sql = ",\n            ".join(value_rows)
    execute_statement(
        f"""
        INSERT INTO {APPROVALS_TABLE}
            (mapping_id, new_status, decided_by, decided_at, committed_at)
        VALUES
            {values_sql}
        """
    )

    count = len(edits)
    discard_edits()
    st.session_state["mappings_df"] = load_full_table()
    return count


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

init_session_state()

try:
    current_user = get_current_user()
except Exception:
    current_user = "unknown"

pending_edits = st.session_state["pending_edits"]

st.title("Column Mapping Review")
st.caption(f"Table: `{MAPPINGS_TABLE}`  |  Warehouse: `{WAREHOUSE_ID}`")

# -- sidebar ---------------------------------------------------------------
st.sidebar.markdown(f"**Signed in as:** `{current_user}`")
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
    st.session_state.pop("mappings_df", None)
    st.rerun()

st.sidebar.header("Filters")

# -- load data --------------------------------------------------------------
try:
    raw_df = get_mappings_df()
except Exception as exc:
    _tb = traceback.format_exc()
    _log(f"data load failed: {exc}\n{_tb}")
    st.error(f"Failed to load data: {exc}")
    with st.expander("Debug info"):
        st.code(_tb)
        st.write({
            "WAREHOUSE_ID": WAREHOUSE_ID or "(not set)",
            "MAPPINGS_TABLE": MAPPINGS_TABLE,
            "DATABRICKS_HOST": os.getenv("DATABRICKS_HOST", "(not set)"),
        })
    st.stop()

display_df = apply_local_edits(raw_df)

all_statuses = ["All"] + sorted(
    display_df["approval_status"].dropna().unique().tolist()
)
all_sources = ["All"] + sorted(display_df["source_system"].dropna().unique().tolist())
all_domains = ["All"] + sorted(display_df["domain"].dropna().unique().tolist())

status_filter = st.sidebar.selectbox("Status", all_statuses, index=0)
source_filter = st.sidebar.selectbox("Source system", all_sources, index=0)
domain_filter = st.sidebar.selectbox("Domain", all_domains, index=0)
min_confidence = st.sidebar.slider("Min confidence", 0, 100, 0)

filtered_df = display_df.copy()
if status_filter != "All":
    filtered_df = filtered_df[filtered_df["approval_status"] == status_filter]
if source_filter != "All":
    filtered_df = filtered_df[filtered_df["source_system"] == source_filter]
if domain_filter != "All":
    filtered_df = filtered_df[filtered_df["domain"] == domain_filter]
filtered_df = filtered_df[filtered_df["confidence_score"] >= min_confidence]

# -- summary metrics --------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total shown", len(filtered_df))
if len(filtered_df):
    col2.metric("Avg confidence", f"{filtered_df['confidence_score'].mean():.0f}")
    pending_count = (filtered_df["approval_status"] == "pending_review").sum()
    col3.metric("Pending review", int(pending_count))
col4.metric("Uncommitted edits", len(pending_edits))

st.divider()

if filtered_df.empty:
    st.info("No mappings match the current filters.")
    st.stop()

# -- data table with row selection ------------------------------------------
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

# -- actions ----------------------------------------------------------------
st.subheader("Actions")

if not selected_ids:
    st.info("Select one or more rows above, then choose an action.")
else:
    st.write(f"**{len(selected_ids)}** mapping(s) selected.")

act_col1, act_col2, act_col3, act_col4, _ = st.columns([1, 1, 1, 1, 3])

with act_col1:
    if st.button("Approve", type="primary", disabled=not selected_ids):
        n = stage_edits(selected_ids, "approved", current_user)
        st.toast(f"Staged {n} approval(s).")
        st.rerun()

with act_col2:
    if st.button("Reject", disabled=not selected_ids):
        n = stage_edits(selected_ids, "rejected", current_user)
        st.toast(f"Staged {n} rejection(s).")
        st.rerun()

with act_col3:
    if st.button(
        "Commit all",
        type="primary",
        disabled=not pending_edits,
        key="main_commit",
    ):
        n = commit_edits()
        st.toast(f"Committed {n} decision(s).")
        st.rerun()

with act_col4:
    if st.button("Discard all", disabled=not pending_edits, key="main_discard"):
        discard_edits()
        st.toast("All edits discarded.")
        st.rerun()

# -- pending edits summary --------------------------------------------------
if pending_edits:
    st.divider()
    st.subheader("Pending edits (uncommitted)")
    edits_df = pd.DataFrame(pending_edits.values())
    st.dataframe(edits_df, use_container_width=True, hide_index=True)

_log("render complete")
