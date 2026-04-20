"""
Column Mapper
=============
Two-page Streamlit app for cross-platform column standardization.

  Data Mapping -- Discover source columns, run the AI mapping agent, review proposals
  Master File  -- Manage the canonical field registry, gold views, and audit log

All tables are Delta tables in Unity Catalog (serverless SQL warehouse).
Tables and seed data are created by the 01_setup_data Databricks job.
"""

import json
import os
import sys
import uuid
from difflib import SequenceMatcher

import pandas as pd
import streamlit as st
from databricks.sdk import WorkspaceClient


def _log(msg):
    print(f"[column-mapper] {msg}", file=sys.stderr, flush=True)


# =====================================================================
# PAGE CONFIG & BRANDING
# =====================================================================

st.set_page_config(
    page_title="Column Mapper",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

section[data-testid="stMainBlockContainer"],
[data-testid="stAppViewContainer"],
.main .block-container { background-color: #ffffff; color: #1a1a1a; }
header[data-testid="stHeader"] { background-color: #ffffff; }

h1, .stMarkdown h1 { color: #1B2A4A !important; font-weight: 700 !important; }
h2, h3, h4, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
    color: #1B2A4A !important; font-weight: 600 !important;
}

[data-testid="stSidebarContent"] { background-color: #f8f9fa; color: #1a1a1a; }
[data-testid="stSidebarContent"] h1,
[data-testid="stSidebarContent"] h2,
[data-testid="stSidebarContent"] h3 { color: #1B2A4A !important; }
[data-testid="stSidebarContent"] label,
[data-testid="stSidebarContent"] .stMarkdown p { color: #1B2A4A !important; }
[data-testid="stSidebarContent"] .stCaption p { color: #555 !important; }

[data-testid="stMetricValue"] { color: #1B2A4A; font-weight: 700; }
[data-testid="stMetricLabel"] p { color: #555; font-weight: 500; }

div[data-testid="stMetric"] {
    background-color: #f8f9fa; border: 1px solid #e0e0e0;
    border-radius: 8px; padding: 12px 16px;
}

.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background-color: #1B2A4A !important; border-color: #1B2A4A !important;
    color: #ffffff !important; font-weight: 600 !important;
    border-radius: 6px !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background-color: #2C3F66 !important; border-color: #2C3F66 !important;
}

hr { border-color: #e0e0e0 !important; opacity: 0.8; }
.stProgress > div > div > div > div { background-color: #1B2A4A; }

[data-testid="stTabs"] [data-baseweb="tab"] {
    font-weight: 500; font-size: 0.95em; padding: 10px 20px;
}
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
    color: #1B2A4A !important; border-bottom-color: #1B2A4A !important;
    font-weight: 600;
}

.badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 0.75em; font-weight: 600; margin: 1px 2px; color: #fff;
}
.badge-high { background-color: #10B981; }
.badge-medium { background-color: #F59E0B; }
.badge-low { background-color: #EF4444; }
.badge-active { background-color: #10B981; }
.badge-pending { background-color: #F59E0B; }
.badge-linked { background-color: #3B82F6; }
.badge-plat   { background-color: #3B82F6; }
.badge-plat-1 { background-color: #10B981; }
.badge-plat-2 { background-color: #8B5CF6; }
.badge-plat-3 { background-color: #EF4444; }
.badge-plat-4 { background-color: #F59E0B; }
.badge-plat-5 { background-color: #EC4899; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# =====================================================================
# CONFIGURATION
# =====================================================================

def _load_config_yaml():
    try:
        import yaml
    except ImportError:
        return {}
    for candidate in [
        os.path.join(os.path.dirname(__file__), "..", "config.yaml"),
        os.path.join(os.path.dirname(__file__), "config.yaml"),
        "config.yaml",
    ]:
        path = os.path.abspath(candidate)
        if os.path.isfile(path):
            with open(path) as f:
                return yaml.safe_load(f) or {}
    return {}


CFG = _load_config_yaml()
DB = CFG.get("databricks", {})
TABLES = CFG.get("tables", {})
PLATFORMS = CFG.get("platforms", [])
GOLD = CFG.get("gold", {})
AGENT_CFG = CFG.get("agent", {})

CATALOG = DB.get("catalog", "column_mapping")
SCHEMA = DB.get("schema", "mapping")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID") or DB.get("warehouse_id", "")
LLM_ENDPOINT = CFG.get("llm", {}).get("endpoint", "databricks-claude-sonnet-4-5")

CONFIDENCE_HIGH_MIN = AGENT_CFG.get("confidence_high_min", 85)
CONFIDENCE_MEDIUM_MIN = AGENT_CFG.get("confidence_medium_min", 60)
DETERMINISTIC_EXACT = AGENT_CFG.get("deterministic_exact_confidence", 95)
DETERMINISTIC_NEAR = AGENT_CFG.get("deterministic_near_confidence", 90)
MAX_LLM_WORKERS = AGENT_CFG.get("max_concurrent_llm_calls", 8)
MAX_RATIONALE_LEN = AGENT_CFG.get("max_rationale_length", 500)

GOLD_CATALOG = GOLD.get("catalog", CATALOG)
GOLD_SCHEMA = GOLD.get("schema", "gold")

PLAT_BY_ID = {p["id"]: p for p in PLATFORMS}
PLAT_NAMES = {p["id"]: p["name"] for p in PLATFORMS}
PLAT_IDS = [p["id"] for p in PLATFORMS]

_BADGE_CLASSES = ["badge-plat", "badge-plat-1", "badge-plat-2", "badge-plat-3", "badge-plat-4", "badge-plat-5"]
PLAT_BADGE_CLASS = {pid: _BADGE_CLASSES[i % len(_BADGE_CLASSES)] for i, pid in enumerate(PLAT_IDS)}


def _fqn(table_key: str) -> str:
    return f"{CATALOG}.{SCHEMA}.{TABLES.get(table_key, table_key)}"


T_CANONICAL = _fqn("canonical_fields")
T_SOURCE = _fqn("source_columns")
T_PROPOSALS = _fqn("mapping_proposals")
T_APPROVED = _fqn("approved_mappings")
T_AUDIT = _fqn("audit_log")
T_RULES = _fqn("standardization_rules")


# =====================================================================
# DATABASE HELPERS (serverless SQL warehouse)
# =====================================================================

@st.cache_resource
def get_ws():
    profile = DB.get("profile")
    if profile:
        return WorkspaceClient(profile=profile)
    return WorkspaceClient()


def get_current_user() -> str:
    try:
        headers = st.context.headers
        for key in ("X-Forwarded-Email", "X-Forwarded-Preferred-Username"):
            if (val := headers.get(key)):
                return val
    except Exception:
        pass
    try:
        me = get_ws().current_user.me()
        return me.user_name or me.display_name or "unknown"
    except Exception:
        return "unknown"


def _get_warehouse_id() -> str:
    global WAREHOUSE_ID
    if WAREHOUSE_ID:
        return WAREHOUSE_ID
    ws = get_ws()
    for wh in ws.warehouses.list():
        if wh.state and wh.state.value in ("RUNNING", "STARTING"):
            WAREHOUSE_ID = wh.id
            return WAREHOUSE_ID
    for wh in ws.warehouses.list():
        if wh.id:
            WAREHOUSE_ID = wh.id
            return WAREHOUSE_ID
    raise RuntimeError("No SQL warehouse found. Set DATABRICKS_WAREHOUSE_ID or add warehouse_id to config.yaml.")


def _esc(val) -> str:
    if val is None:
        return "NULL"
    return "'" + str(val).replace("'", "''") + "'"


def run_sql(query: str) -> list[dict]:
    """Execute SQL via the serverless warehouse and return rows as dicts."""
    wid = _get_warehouse_id()
    resp = get_ws().statement_execution.execute_statement(
        warehouse_id=wid, statement=query, wait_timeout="50s",
    )
    if resp.status and resp.status.error:
        raise RuntimeError(f"SQL error: {resp.status.error.message}")
    if resp.result and resp.result.data_array:
        cols = [c.name for c in resp.manifest.schema.columns]
        return [dict(zip(cols, row)) for row in resp.result.data_array]
    return []


def run_stmt(query: str) -> None:
    """Execute a DML statement (INSERT/UPDATE/DELETE) via the SQL warehouse."""
    wid = _get_warehouse_id()
    resp = get_ws().statement_execution.execute_statement(
        warehouse_id=wid, statement=query, wait_timeout="50s",
    )
    if resp.status and resp.status.error:
        raise RuntimeError(f"SQL error: {resp.status.error.message}")


# =====================================================================
# CACHED DATA LOADING
# =====================================================================

def _safe_load(query: str) -> list[dict]:
    try:
        return run_sql(query)
    except RuntimeError as exc:
        msg = str(exc)
        if "TABLE_OR_VIEW_NOT_FOUND" in msg or "UNRESOLVED_COLUMN" in msg:
            _log(f"safe_load fallback: {msg[:120]}")
            return []
        raise


@st.cache_data(ttl=120, show_spinner=False)
def load_canonical():
    rows = _safe_load(f"SELECT * FROM {T_CANONICAL} ORDER BY canonical_name")
    return [r for r in rows if r.get("is_active") in (True, "true", None)]


@st.cache_data(ttl=120, show_spinner=False)
def load_source_columns():
    return _safe_load(f"SELECT * FROM {T_SOURCE} ORDER BY platform_id, source_table, column_name")


@st.cache_data(ttl=60, show_spinner=False)
def load_proposals():
    return _safe_load(f"SELECT * FROM {T_PROPOSALS} ORDER BY confidence DESC")


@st.cache_data(ttl=120, show_spinner=False)
def load_approved_mappings():
    return _safe_load(f"SELECT * FROM {T_APPROVED} ORDER BY approved_at DESC")


@st.cache_data(ttl=60, show_spinner=False)
def load_audit(limit=500):
    return _safe_load(f"SELECT * FROM {T_AUDIT} ORDER BY created_at DESC LIMIT {limit}")


@st.cache_data(ttl=300, show_spinner=False)
def load_rules():
    return _safe_load(f"SELECT * FROM {T_RULES} ORDER BY rule_type, pattern")


def invalidate():
    for fn in [load_canonical, load_source_columns, load_proposals,
               load_approved_mappings, load_audit, load_rules]:
        fn.clear()


if "cache_warmed" not in st.session_state:
    invalidate()
    st.session_state["cache_warmed"] = True


# =====================================================================
# AUDIT LOGGING
# =====================================================================

def log_audit(entity_type: str, entity_id: str, action: str, actor: str,
              details: dict | None = None):
    eid = uuid.uuid4().hex[:12]
    run_stmt(
        f"INSERT INTO {T_AUDIT} "
        f"(event_id, entity_type, entity_id, action, actor, details, created_at) "
        f"VALUES ({_esc(eid)}, {_esc(entity_type)}, {_esc(entity_id)}, "
        f"{_esc(action)}, {_esc(actor)}, "
        f"{_esc(json.dumps(details) if details else None)}, current_timestamp())"
    )


# =====================================================================
# BATCH MAPPING JOB (runs in-app via SQL warehouse)
# =====================================================================

def discover_source_columns() -> int:
    """Discover new columns from INFORMATION_SCHEMA and insert into source_columns."""
    batch_id = uuid.uuid4().hex[:12]

    existing = _safe_load(f"SELECT platform_id, source_table, column_name FROM {T_SOURCE}")
    existing_set = {(r["platform_id"], r["source_table"], r["column_name"]) for r in existing}

    new_rows = []
    for plat in PLATFORMS:
        pid = plat["id"]
        src_catalog = plat.get("source_catalog", CATALOG)
        src_schema = plat.get("source_schema", pid)
        try:
            cols = run_sql(
                f"SELECT table_catalog, table_schema, table_name, column_name, data_type "
                f"FROM {src_catalog}.information_schema.columns "
                f"WHERE table_schema = '{src_schema}'"
            )
        except Exception as exc:
            _log(f"discover skip {pid}: {exc}")
            continue

        for col in cols:
            fq_table = f"{col['table_catalog']}.{col['table_schema']}.{col['table_name']}"
            key = (pid, fq_table, col["column_name"])
            if key not in existing_set:
                cid = uuid.uuid4().hex[:12]
                new_rows.append(
                    f"({_esc(cid)}, {_esc(pid)}, {_esc(fq_table)}, "
                    f"{_esc(col['column_name'])}, {_esc(col.get('data_type', 'STRING'))}, "
                    f"{_esc(batch_id)}, current_timestamp())"
                )
                existing_set.add(key)

    batch_size = 100
    for i in range(0, len(new_rows), batch_size):
        chunk = new_rows[i : i + batch_size]
        run_stmt(
            f"INSERT INTO {T_SOURCE} "
            f"(column_id, platform_id, source_table, column_name, data_type, batch_id, detected_at) "
            f"VALUES {', '.join(chunk)}"
        )

    return len(new_rows)


def _ensure_src_on_path():
    """Add the src directory to sys.path for importing column_mapping modules."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "src"),
        os.path.join(os.path.dirname(__file__), "src"),
    ]
    for candidate in candidates:
        abs_path = os.path.abspath(candidate)
        if os.path.isdir(abs_path) and abs_path not in sys.path:
            sys.path.insert(0, abs_path)
            return


def _deterministic_fast_match(col: dict, canonical: list[dict], rules: list[dict]) -> dict | None:
    """Try to match a column to a canonical field without calling the LLM.

    Returns a result dict if there's a high-confidence deterministic match,
    or None to fall back to the full LLM pipeline.
    """
    from column_mapping.agent_tools import deterministic_standardize, get_abbreviation_rules

    abbrevs = get_abbreviation_rules(rules)
    std_name = deterministic_standardize(col["column_name"], abbrevs)

    canon_by_name = {c["canonical_name"]: c for c in canonical}
    if std_name in canon_by_name:
        c = canon_by_name[std_name]
        return {
            "canonical_id": c["canonical_id"],
            "canonical_name": c["canonical_name"],
            "confidence": DETERMINISTIC_EXACT,
            "rationale": f"Deterministic match: '{col['column_name']}' standardizes to '{std_name}' which exactly matches canonical field.",
        }

    for cname, c in canon_by_name.items():
        if std_name.replace("_", "") == cname.replace("_", ""):
            return {
                "canonical_id": c["canonical_id"],
                "canonical_name": c["canonical_name"],
                "confidence": DETERMINISTIC_NEAR,
                "rationale": f"Near-exact match: '{col['column_name']}' standardizes to '{std_name}', matches '{cname}' after normalization.",
            }

    return None


def run_batch_mapping(progress_bar=None, status_text=None) -> int:
    """Run the mapping agent on all unmapped source columns.

    Uses a deterministic fast-path for obvious matches and parallel LLM
    calls (8 concurrent threads) for the rest.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _ensure_src_on_path()

    from column_mapping.mapping_agent import run_mapping_agent

    batch_id = uuid.uuid4().hex[:12]
    all_source = _safe_load(f"SELECT * FROM {T_SOURCE}")
    approved = _safe_load(f"SELECT * FROM {T_APPROVED}")
    canonical = _safe_load(f"SELECT * FROM {T_CANONICAL}")
    rules = _safe_load(f"SELECT * FROM {T_RULES}")

    mapped_ids = {m["column_id"] for m in approved}
    proposed_rows = _safe_load(
        f"SELECT column_id FROM {T_PROPOSALS} WHERE status IN ('pending_review', 'approved')"
    )
    proposed_ids = {r["column_id"] for r in proposed_rows}

    unmapped = [
        c for c in all_source
        if c["column_id"] not in mapped_ids and c["column_id"] not in proposed_ids
    ]

    if not unmapped:
        return 0

    counter = {"done": 0, "written": 0}
    lock = threading.Lock()
    total = len(unmapped)

    def _write_proposal(col, canonical_id, canonical_name, confidence, rationale, model_name):
        confidence_level = (
            "high" if confidence >= CONFIDENCE_HIGH_MIN
            else "medium" if confidence >= CONFIDENCE_MEDIUM_MIN
            else "low"
        )
        prop_id = uuid.uuid4().hex[:12]
        audit_id = uuid.uuid4().hex[:12]
        run_stmt(
            f"INSERT INTO {T_PROPOSALS} "
            f"(proposal_id, column_id, suggested_canonical_id, suggested_canonical_name, "
            f"confidence, confidence_level, reasoning, agent_model, batch_id, status, created_at) "
            f"VALUES ({_esc(prop_id)}, {_esc(col['column_id'])}, "
            f"{_esc(canonical_id)}, {_esc(canonical_name)}, "
            f"{confidence}, {_esc(confidence_level)}, "
            f"{_esc((rationale or '')[:MAX_RATIONALE_LEN])}, {_esc(model_name)}, "
            f"{_esc(batch_id)}, 'pending_review', current_timestamp());"
            f"INSERT INTO {T_AUDIT} "
            f"(event_id, entity_type, entity_id, action, actor, details, created_at) "
            f"VALUES ({_esc(audit_id)}, 'proposal', {_esc(prop_id)}, "
            f"'created', 'agent_batch', "
            f"{_esc(json.dumps({'column': col['column_name'], 'confidence': confidence}))}, "
            f"current_timestamp())"
        )
        return True

    # Phase 1: deterministic fast-path (no LLM needed)
    need_llm = []
    for col in unmapped:
        fast = _deterministic_fast_match(col, canonical, rules)
        if fast:
            _write_proposal(col, fast["canonical_id"], fast["canonical_name"],
                            fast["confidence"], fast["rationale"], "deterministic")
            counter["written"] += 1
            counter["done"] += 1
            _log(f"fast-match: {col['column_name']} -> {fast['canonical_name']}")
        else:
            need_llm.append(col)

    if status_text:
        status_text.text(f"Fast-matched {counter['written']}/{total}. Running LLM on {len(need_llm)} remaining...")
    if progress_bar:
        progress_bar.progress(counter["done"] / total if total else 1.0)

    # Phase 2: parallel LLM calls for remaining columns
    def _process_one(col):
        pname = PLAT_NAMES.get(col["platform_id"], col["platform_id"])
        try:
            result = run_mapping_agent(
                column_name=col["column_name"],
                platform_id=col["platform_id"],
                platform_name=pname,
                source_columns=all_source,
                approved_mappings=approved,
                canonical_fields=canonical,
                rules=rules,
                sql_fn=run_sql,
                llm_endpoint=LLM_ENDPOINT,
            )
        except Exception as exc:
            _log(f"agent error for {col['column_name']}: {exc}")
            return None

        if result.error:
            _log(f"agent result error for {col['column_name']}: {result.error}")
            return None

        _write_proposal(
            col, result.recommended_canonical_id, result.recommended_canonical_name,
            result.confidence, result.rationale, LLM_ENDPOINT,
        )
        return col["column_name"]

    if need_llm:
        with ThreadPoolExecutor(max_workers=MAX_LLM_WORKERS) as pool:
            futures = {pool.submit(_process_one, col): col for col in need_llm}
            for future in as_completed(futures):
                col = futures[future]
                with lock:
                    counter["done"] += 1
                    name = future.result()
                    if name:
                        counter["written"] += 1
                    if progress_bar:
                        progress_bar.progress(counter["done"] / total)
                    if status_text:
                        status_text.text(
                            f"[{counter['done']}/{total}] "
                            f"{PLAT_NAMES.get(col['platform_id'], '')} | {col['column_name']}"
                        )

    return counter["written"]


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def name_similarity(name_a: str, name_b: str) -> float:
    if not name_a or not name_b:
        return 0.0
    tokens_a = set(name_a.lower().split("_"))
    tokens_b = set(name_b.lower().split("_"))
    overlap = len(tokens_a & tokens_b) / max(len(tokens_a), len(tokens_b))
    sequence = SequenceMatcher(None, name_a.lower(), name_b.lower()).ratio()
    return overlap * 0.5 + sequence * 0.5


def find_similar_canonicals(name: str, canonical_list: list[dict], threshold: float = 0.6) -> list[tuple[str, float]]:
    similar = []
    for canon in canonical_list:
        sim = name_similarity(name, canon.get("canonical_name", ""))
        if sim >= threshold:
            similar.append((canon["canonical_name"], round(sim, 2)))
    return sorted(similar, key=lambda x: x[1], reverse=True)


def platform_badge_html(platform_id: str, count: int | None = None) -> str:
    name = PLAT_NAMES.get(platform_id, platform_id)
    cls = PLAT_BADGE_CLASS.get(platform_id, "badge-plat")
    label = f"{name} {count}" if count is not None else name
    return f'<span class="badge {cls}">{label}</span>'


def confidence_badge_html(level: str) -> str:
    cls_map = {"high": "badge-high", "medium": "badge-medium", "low": "badge-low"}
    cls = cls_map.get((level or "").lower(), "badge-low")
    return f'<span class="badge {cls}">{(level or "unknown").title()}</span>'


def status_badge_html(status: str) -> str:
    s = (status or "").lower()
    cls_map = {"active": "badge-active", "pending": "badge-pending", "linked": "badge-linked"}
    cls = cls_map.get(s, "badge-plat")
    return f'<span class="badge {cls}">{(status or "unknown").title()}</span>'


def build_cross_platform_context(canonical_id: str, approved: list[dict],
                                  source_cols: list[dict]) -> dict[str, list[str]]:
    col_lookup = {c["column_id"]: c for c in source_cols}
    result: dict[str, list[str]] = {}
    for m in approved:
        if m.get("canonical_id") != canonical_id:
            continue
        col = col_lookup.get(m.get("column_id"), {})
        pid = col.get("platform_id", "")
        cname = col.get("column_name", "")
        if pid and cname:
            result.setdefault(pid, []).append(cname)
    return result


def generate_gold_view_sql(source_table: str, mappings: list[dict],
                            source_cols: list[dict], canonical_list: list[dict],
                            platform_id: str) -> str | None:
    col_lookup = {c["column_id"]: c for c in source_cols}
    canon_lookup = {c["canonical_id"]: c for c in canonical_list}

    rename_pairs = []
    for m in mappings:
        col = col_lookup.get(m.get("column_id"), {})
        canon = canon_lookup.get(m.get("canonical_id"), {})
        if col.get("source_table") == source_table and col.get("column_name") and canon.get("canonical_name"):
            rename_pairs.append((col["column_name"], canon["canonical_name"]))

    if not rename_pairs:
        return None

    safe_name = source_table.split(".")[-1]
    view_name = f"{GOLD_CATALOG}.{GOLD_SCHEMA}.{platform_id}__{safe_name}"

    select_parts = [f"    `{orig}` AS `{canonical}`" for orig, canonical in rename_pairs]
    select_clause = ",\n".join(select_parts)

    return f"CREATE OR REPLACE VIEW {view_name} AS\nSELECT\n{select_clause}\nFROM {source_table}"


# =====================================================================
# SIDEBAR
# =====================================================================

try:
    current_user = get_current_user()
except Exception:
    current_user = "unknown"

st.sidebar.markdown(
    '<div style="text-align:center; padding:0.5rem 0;">'
    '<span style="font-size:1.1em; font-weight:600; color:#1B2A4A;">Column Mapper</span></div>',
    unsafe_allow_html=True,
)
st.sidebar.divider()
st.sidebar.caption(f"Signed in as **{current_user}**")
st.sidebar.caption(f"`{CATALOG}`.`{SCHEMA}`")
st.sidebar.caption(f"{len(PLATFORMS)} platforms configured")

st.sidebar.divider()

if st.sidebar.button("Reload data"):
    invalidate()
    st.rerun()

# =====================================================================
# TOP TAB NAVIGATION
# =====================================================================
tab_mapping, tab_approved, tab_master = st.tabs(["Data Mapping", "Approved Mappings", "Master File Management"])

# =====================================================================
# PAGE 1: DATA MAPPING
# =====================================================================
with tab_mapping:
    st.markdown("## Data Mapping")

    if "auto_discovered" not in st.session_state:
        with st.spinner("Discovering source columns from platform schemas..."):
            discover_source_columns()
        invalidate()
        st.session_state["auto_discovered"] = True

    proposals = load_proposals()
    source_cols = load_source_columns()
    approved = load_approved_mappings()
    canonical_raw = load_canonical()

    col_lookup = {c["column_id"]: c for c in source_cols}
    canon_lookup = {c["canonical_id"]: c for c in canonical_raw}

    pending = [p for p in proposals if p.get("status") == "pending_review"]
    approved_proposals = [p for p in proposals if p.get("status") == "approved"]

    total_proposals = len(proposals)
    total_approved = len(approved_proposals)
    proposed_ids = {p.get("column_id") for p in proposals}
    unmapped_for_agent = len([c for c in source_cols if c["column_id"] not in proposed_ids])

    # -- Summary metrics --
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Source Columns", len(source_cols))
    s2.metric("Unmapped", unmapped_for_agent)
    s3.metric("Pending Review", len(pending))
    s4.metric("Approved", total_approved)

    # ----------------------------------------------------------------
    # STEP 1: Run the Mapping
    # ----------------------------------------------------------------
    st.divider()
    st.markdown("### Step 1 -- Run the Mapping")
    st.caption(
        "Discover columns from platform schemas, then run the AI agent "
        "to propose canonical matches for unmapped columns."
    )

    plat_col_counts: dict[str, dict] = {}
    for c in source_cols:
        pid = c.get("platform_id", "")
        tbl = c.get("source_table", "").split(".")[-1] if c.get("source_table") else ""
        plat_col_counts.setdefault(pid, {"columns": 0, "tables": set()})
        plat_col_counts[pid]["columns"] += 1
        if tbl:
            plat_col_counts[pid]["tables"].add(tbl)

    if plat_col_counts:
        plat_rows = []
        for plat in PLATFORMS:
            pid = plat["id"]
            info = plat_col_counts.get(pid, {"columns": 0, "tables": set()})
            proposed_for_plat = len([
                p for p in proposals
                if col_lookup.get(p.get("column_id"), {}).get("platform_id") == pid
            ])
            approved_for_plat = len([
                m for m in approved
                if col_lookup.get(m.get("column_id"), {}).get("platform_id") == pid
            ])
            plat_rows.append({
                "Platform": plat["name"],
                "Description": plat.get("description", ""),
                "Schema": f"{plat.get('source_catalog', CATALOG)}.{plat.get('source_schema', pid)}",
                "Tables": ", ".join(sorted(info["tables"])) if info["tables"] else "--",
                "Columns": info["columns"],
                "Proposals": proposed_for_plat,
                "Approved": approved_for_plat,
            })
        st.dataframe(pd.DataFrame(plat_rows), use_container_width=True, hide_index=True)

    bc1, bc2, bc3 = st.columns([1, 1, 3])
    with bc1:
        if st.button("Discover Columns", type="primary", key="discover_btn"):
            with st.spinner("Scanning INFORMATION_SCHEMA..."):
                new_count = discover_source_columns()
            invalidate()
            if new_count > 0:
                st.toast(f"Discovered {new_count} new source column(s).")
            else:
                st.toast("No new columns found (all already discovered).")
            st.rerun()
    with bc2:
        if st.button(
            f"Run Mapping Agent ({unmapped_for_agent} unmapped)",
            type="primary",
            disabled=len(source_cols) == 0,
            key="run_agent_btn",
        ):
            progress_bar = st.progress(0)
            status_text = st.empty()
            with st.spinner("Running AI mapping agent..."):
                written = run_batch_mapping(progress_bar=progress_bar, status_text=status_text)
            invalidate()
            progress_bar.empty()
            status_text.empty()
            if written > 0:
                st.toast(f"Created {written} mapping proposal(s).")
            else:
                st.toast("No new proposals needed (all columns already have proposals).")
            st.rerun()

    # ----------------------------------------------------------------
    # STEP 2: Review the Mapping
    # ----------------------------------------------------------------
    st.divider()
    st.markdown("### Step 2 -- Review the Mapping")
    st.caption(
        "Filter and review AI proposals. Select rows to approve, reject, "
        "flag, or reassign to a different canonical field."
    )

    fc1, fc2, fc3 = st.columns([1, 1, 2])
    with fc1:
        wq_plat = st.selectbox(
            "Platform", ["All"] + PLAT_IDS,
            format_func=lambda x: "All Platforms" if x == "All" else PLAT_NAMES.get(x, x),
            key="wq_plat",
        )
    with fc2:
        wq_status = st.selectbox(
            "Status",
            ["pending_review", "All", "flagged", "approved", "rejected"],
            format_func=lambda x: x.replace("_", " ").title(),
            key="wq_status",
        )
    with fc3:
        wq_search = st.text_input(
            "Search across headers, tables, notes...",
            key="wq_search", placeholder="Search...",
        )

    display_proposals = proposals if wq_status == "All" else [p for p in proposals if p.get("status") == wq_status]
    if wq_plat != "All":
        display_proposals = [
            p for p in display_proposals
            if col_lookup.get(p.get("column_id"), {}).get("platform_id") == wq_plat
        ]
    if wq_search:
        q = wq_search.lower()
        display_proposals = [
            p for p in display_proposals
            if q in (col_lookup.get(p.get("column_id"), {}).get("column_name", "")).lower()
            or q in (col_lookup.get(p.get("column_id"), {}).get("source_table", "")).lower()
            or q in (canon_lookup.get(p.get("suggested_canonical_id"), {}).get("canonical_name", "")).lower()
            or q in (p.get("reasoning") or "").lower()
        ]

    sel_proposal_ids = []

    if not display_proposals:
        if total_proposals == 0 and len(source_cols) == 0:
            st.info("No source columns discovered yet. Run Step 1 above to get started.")
        elif total_proposals == 0:
            st.info("Source columns discovered but no proposals yet. Run the Mapping Agent in Step 1.")
        else:
            st.info("No proposals match the current filters.")
    else:
        rows = []
        for p in display_proposals:
            col = col_lookup.get(p.get("column_id"), {})
            canon = canon_lookup.get(p.get("suggested_canonical_id"), {})

            cross_plat = {}
            if p.get("suggested_canonical_id"):
                cross_plat = build_cross_platform_context(
                    p["suggested_canonical_id"], approved, source_cols
                )

            row = {
                "proposal_id": p.get("proposal_id", ""),
                "Suggested Match": canon.get("canonical_name", p.get("suggested_canonical_name", "")),
                "Confidence": p.get("confidence_level", "").title(),
                "Header": col.get("column_name", ""),
                "Platform": PLAT_NAMES.get(col.get("platform_id", ""), col.get("platform_id", "")),
                "Source Table": (col.get("source_table", "").split(".")[-1] if col.get("source_table") else ""),
                "Status": (p.get("status") or "").replace("_", " ").title(),
            }
            for pid in PLAT_IDS:
                pname = PLAT_NAMES[pid]
                mapped_cols = cross_plat.get(pid, [])
                row[pname] = ", ".join(mapped_cols) if mapped_cols else ""

            rows.append(row)

        wq_df = pd.DataFrame(rows)

        display_cols = ["Suggested Match", "Confidence", "Header", "Platform", "Source Table"]
        platform_cols = [PLAT_NAMES[pid] for pid in PLAT_IDS]
        display_cols += platform_cols
        display_cols.append("Status")

        show_cols = [c for c in display_cols if c in wq_df.columns]
        show_df = wq_df[show_cols]

        event = st.dataframe(
            show_df, use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="multi-row",
        )
        sel_rows = event.selection.rows if event.selection else []
        sel_proposal_ids = [wq_df.iloc[i]["proposal_id"] for i in sel_rows]

        if sel_proposal_ids:
            st.markdown(
                f'<div style="background:#1B2A4A; color:white; padding:8px 16px; '
                f'border-radius:6px; margin:8px 0;">'
                f'<strong>{len(sel_proposal_ids)} item(s) selected</strong></div>',
                unsafe_allow_html=True,
            )

        ac1, ac2, ac3, ac4 = st.columns([1, 1, 1, 4])
        with ac1:
            if st.button("Approve Selected", type="primary", disabled=not sel_proposal_ids, key="wq_approve"):
                for pid in sel_proposal_ids:
                    prop = next((p for p in display_proposals if p["proposal_id"] == pid), None)
                    if not prop or not prop.get("suggested_canonical_id"):
                        continue
                    mid = uuid.uuid4().hex[:12]
                    run_stmt(
                        f"INSERT INTO {T_APPROVED} "
                        f"(mapping_id, column_id, canonical_id, proposal_id, approved_by, approved_at) "
                        f"VALUES ({_esc(mid)}, {_esc(prop['column_id'])}, "
                        f"{_esc(prop['suggested_canonical_id'])}, {_esc(pid)}, "
                        f"{_esc(current_user)}, current_timestamp())"
                    )
                    run_stmt(
                        f"UPDATE {T_PROPOSALS} SET status = 'approved', "
                        f"reviewed_at = current_timestamp(), reviewed_by = {_esc(current_user)} "
                        f"WHERE proposal_id = {_esc(pid)}"
                    )
                    log_audit("mapping", mid, "approved", current_user,
                              {"canonical_id": prop["suggested_canonical_id"], "column_id": prop["column_id"]})
                invalidate()
                st.toast(f"Approved {len(sel_proposal_ids)} mapping(s).")
                st.rerun()
        with ac2:
            if st.button("Flag for Discussion", disabled=not sel_proposal_ids, key="wq_flag"):
                for pid in sel_proposal_ids:
                    run_stmt(
                        f"UPDATE {T_PROPOSALS} SET status = 'flagged' "
                        f"WHERE proposal_id = {_esc(pid)}"
                    )
                    log_audit("proposal", pid, "flagged", current_user)
                invalidate()
                st.toast(f"Flagged {len(sel_proposal_ids)} item(s) for discussion.")
                st.rerun()
        with ac3:
            if st.button("Reject Selected", disabled=not sel_proposal_ids, key="wq_reject"):
                for pid in sel_proposal_ids:
                    run_stmt(
                        f"UPDATE {T_PROPOSALS} SET status = 'rejected', "
                        f"reviewed_at = current_timestamp(), reviewed_by = {_esc(current_user)} "
                        f"WHERE proposal_id = {_esc(pid)}"
                    )
                    log_audit("proposal", pid, "rejected", current_user)
                invalidate()
                st.toast(f"Rejected {len(sel_proposal_ids)} item(s).")
                st.rerun()

        # -- Detail panel for single selection --
        if len(sel_proposal_ids) == 1:
            sel_prop = next((p for p in display_proposals if p["proposal_id"] == sel_proposal_ids[0]), None)
            if sel_prop:
                sel_col = col_lookup.get(sel_prop.get("column_id"), {})
                sel_canon = canon_lookup.get(sel_prop.get("suggested_canonical_id"), {})

                with st.container(border=True):
                    st.markdown("**Proposal Detail**")

                    dc1, dc2, dc3 = st.columns(3)
                    dc1.markdown(f"**Source:** `{sel_col.get('column_name', 'N/A')}` from `{sel_col.get('source_table', 'N/A')}`")
                    dc2.markdown(f"**Suggested Match:** `{sel_canon.get('canonical_name', sel_prop.get('suggested_canonical_name', 'N/A'))}`")
                    try:
                        conf_val = float(sel_prop.get('confidence', 0))
                    except (TypeError, ValueError):
                        conf_val = 0
                    dc3.markdown(f"**Confidence:** {sel_prop.get('confidence_level', 'N/A').title()} ({conf_val:.0f}%)")

                    if sel_prop.get("reasoning"):
                        st.markdown(f"**Agent Reasoning:** {sel_prop['reasoning']}")

                    st.markdown("**Reassign to a different canonical field:**")
                    canon_options = {c["canonical_id"]: c["canonical_name"] for c in canonical_raw}
                    default_cid = sel_prop.get("suggested_canonical_id")
                    cid_list = list(canon_options.keys())
                    default_idx = cid_list.index(default_cid) if default_cid in cid_list else 0

                    chosen_cid = st.selectbox(
                        "Canonical field",
                        cid_list,
                        index=default_idx if cid_list else 0,
                        format_func=lambda x: canon_options.get(x, x),
                        key="wq_reassign",
                    ) if cid_list else None

                    if chosen_cid and st.button("Approve with Selection", type="primary", key="wq_approve_reassign"):
                        mid = uuid.uuid4().hex[:12]
                        run_stmt(
                            f"INSERT INTO {T_APPROVED} "
                            f"(mapping_id, column_id, canonical_id, proposal_id, approved_by, approved_at) "
                            f"VALUES ({_esc(mid)}, {_esc(sel_prop['column_id'])}, "
                            f"{_esc(chosen_cid)}, {_esc(sel_prop['proposal_id'])}, "
                            f"{_esc(current_user)}, current_timestamp())"
                        )
                        run_stmt(
                            f"UPDATE {T_PROPOSALS} SET status = 'approved', "
                            f"reviewed_at = current_timestamp(), reviewed_by = {_esc(current_user)} "
                            f"WHERE proposal_id = {_esc(sel_prop['proposal_id'])}"
                        )
                        reassigned = chosen_cid != default_cid
                        log_audit("mapping", mid, "approved", current_user,
                                  {"canonical_id": chosen_cid, "reassigned": reassigned})
                        invalidate()
                        st.toast("Approved" + (" (reassigned)" if reassigned else "") + ".")
                        st.rerun()

    # ----------------------------------------------------------------
    # STEP 3: Create New Canonical Fields
    # ----------------------------------------------------------------
    st.divider()
    st.markdown("### Step 3 -- Create New Canonical Fields")
    st.caption(
        "If the agent couldn't find a good match, create a new canonical field "
        "and optionally link it to the selected proposal."
    )

    sel_prop_for_create = None
    sel_col_for_create = {}
    if len(sel_proposal_ids) == 1 and display_proposals:
        sel_prop_for_create = next(
            (p for p in display_proposals if p["proposal_id"] == sel_proposal_ids[0]), None
        )
        if sel_prop_for_create:
            sel_col_for_create = col_lookup.get(sel_prop_for_create.get("column_id"), {})

    if sel_prop_for_create:
        st.info(
            f"Creating a new field for: **{sel_col_for_create.get('column_name', '')}** "
            f"({PLAT_NAMES.get(sel_col_for_create.get('platform_id', ''), '')}). "
            f"The new field will be linked to this proposal automatically."
        )
        default_new_name = sel_col_for_create.get("column_name", "").lower().replace(" ", "_")
    else:
        st.info("Select a single proposal in Step 2 above to auto-link, or create a standalone entry.")
        default_new_name = ""

    nc1, nc2, nc3 = st.columns(3)
    with nc1:
        new_name = st.text_input("Standardized Name", value=default_new_name, key="wq_new_name")
    with nc2:
        new_type = st.selectbox("Data Type", ["string", "integer", "decimal", "date", "timestamp", "boolean"], key="wq_new_type")
    with nc3:
        new_domain = st.selectbox("Domain", ["Financial", "Identity", "Temporal", "Operational", "Geographic", "Reference", "Organizational"], key="wq_new_domain")
    new_def = st.text_input("Business Definition", key="wq_new_def", placeholder="e.g. Unique identifier for a fund entity")

    if new_name and canonical_raw:
        similar = find_similar_canonicals(new_name, canonical_raw, threshold=0.6)
        if similar:
            st.warning("Similar entries already exist: " + ", ".join(f"**{s[0]}** ({s[1]:.0%})" for s in similar[:5]))

    create_label = "Create and Link" if sel_prop_for_create else "Create Entry"
    if st.button(create_label, type="primary", disabled=not new_name, key="wq_create_link"):
        cid = uuid.uuid4().hex[:12]
        run_stmt(
            f"INSERT INTO {T_CANONICAL} "
            f"(canonical_id, canonical_name, data_type, business_definition, "
            f"domain_category, is_active, created_by, created_at, updated_at) "
            f"VALUES ({_esc(cid)}, {_esc(new_name)}, {_esc(new_type)}, "
            f"{_esc(new_def)}, {_esc(new_domain)}, true, "
            f"{_esc(current_user)}, current_timestamp(), current_timestamp())"
        )
        log_audit("canonical", cid, "created", current_user, {"name": new_name})

        if sel_prop_for_create:
            mid = uuid.uuid4().hex[:12]
            run_stmt(
                f"INSERT INTO {T_APPROVED} "
                f"(mapping_id, column_id, canonical_id, proposal_id, approved_by, approved_at) "
                f"VALUES ({_esc(mid)}, {_esc(sel_prop_for_create['column_id'])}, "
                f"{_esc(cid)}, {_esc(sel_prop_for_create['proposal_id'])}, "
                f"{_esc(current_user)}, current_timestamp())"
            )
            run_stmt(
                f"UPDATE {T_PROPOSALS} SET status = 'approved', "
                f"reviewed_at = current_timestamp(), reviewed_by = {_esc(current_user)} "
                f"WHERE proposal_id = {_esc(sel_prop_for_create['proposal_id'])}"
            )
            log_audit("mapping", mid, "approved", current_user,
                      {"canonical_id": cid, "new_entry": True})
            st.toast(f"Created '{new_name}' and linked mapping.")
        else:
            st.toast(f"Created canonical field '{new_name}'.")

        invalidate()
        st.rerun()


# =====================================================================
# PAGE 2: APPROVED MAPPINGS (matrix view)
# =====================================================================
with tab_approved:
    st.markdown("## Approved Mappings")
    st.caption(
        "Matrix view of all approved mappings: canonical fields vs. source systems. "
        "Each cell shows the source column name that maps to that canonical field."
    )

    _ap_approved = load_approved_mappings()
    _ap_source = load_source_columns()
    _ap_canonical = load_canonical()

    if not _ap_approved:
        st.info("No approved mappings yet. Approve proposals in the Data Mapping tab to populate this view.")
    else:
        _ap_col_lookup = {c["column_id"]: c for c in _ap_source}
        _ap_canon_lookup = {c["canonical_id"]: c for c in _ap_canonical}

        # Build {canonical_id -> {platform_id -> [column_names]}}
        matrix_data: dict[str, dict[str, list[str]]] = {}
        for m in _ap_approved:
            cid = m.get("canonical_id", "")
            col = _ap_col_lookup.get(m.get("column_id"))
            if not col or not cid:
                continue
            pid = col.get("platform_id", "")
            cname = col.get("column_name", "")
            matrix_data.setdefault(cid, {}).setdefault(pid, []).append(cname)

        # Filters
        af1, af2 = st.columns([2, 1])
        with af1:
            ap_search = st.text_input(
                "Search canonical or source column names...",
                key="ap_search", placeholder="Search...",
            )
        with af2:
            ap_plat_filter = st.selectbox(
                "Platform",
                ["All"] + PLAT_IDS,
                format_func=lambda x: "All Platforms" if x == "All" else PLAT_NAMES.get(x, x),
                key="ap_plat_filter",
            )

        # Build rows: one per canonical field that has at least one mapping
        matrix_rows = []
        for cid, plat_map in sorted(
            matrix_data.items(),
            key=lambda kv: _ap_canon_lookup.get(kv[0], {}).get("canonical_name", ""),
        ):
            canon = _ap_canon_lookup.get(cid, {})
            canon_name = canon.get("canonical_name", cid)

            row: dict[str, str] = {
                "Canonical Field": canon_name,
                "Domain": canon.get("domain_category", ""),
                "Type": canon.get("data_type", ""),
            }
            for pid in PLAT_IDS:
                pname = PLAT_NAMES[pid]
                cols_for_plat = plat_map.get(pid, [])
                row[pname] = ", ".join(cols_for_plat) if cols_for_plat else ""

            row["Coverage"] = f"{len(plat_map)}/{len(PLATFORMS)}"
            matrix_rows.append(row)

        if ap_search:
            q = ap_search.lower()
            matrix_rows = [
                r for r in matrix_rows
                if q in r["Canonical Field"].lower()
                or any(q in r.get(PLAT_NAMES[pid], "").lower() for pid in PLAT_IDS)
            ]

        if ap_plat_filter != "All":
            pname = PLAT_NAMES[ap_plat_filter]
            matrix_rows = [r for r in matrix_rows if r.get(pname, "")]

        # Metrics
        total_canonical_mapped = len(matrix_data)
        total_links = sum(
            sum(len(cols) for cols in plat_map.values())
            for plat_map in matrix_data.values()
        )
        plat_coverage = {
            pid: sum(1 for pm in matrix_data.values() if pid in pm)
            for pid in PLAT_IDS
        }

        am1, am2, am3, am4 = st.columns(4)
        am1.metric("Canonical Fields Mapped", total_canonical_mapped)
        am2.metric("Total Links", total_links)
        am3.metric("Platforms", len(PLATFORMS))
        best_plat = max(plat_coverage, key=plat_coverage.get) if plat_coverage else ""
        am4.metric("Best Coverage", f"{PLAT_NAMES.get(best_plat, '')} ({plat_coverage.get(best_plat, 0)})" if best_plat else "N/A")

        st.divider()

        if not matrix_rows:
            st.info("No mappings match the current filters.")
        else:
            matrix_df = pd.DataFrame(matrix_rows)
            display_cols = ["Canonical Field", "Domain", "Type"]
            display_cols += [PLAT_NAMES[pid] for pid in PLAT_IDS]
            display_cols.append("Coverage")
            show_cols = [c for c in display_cols if c in matrix_df.columns]

            st.dataframe(
                matrix_df[show_cols],
                use_container_width=True,
                hide_index=True,
                height=min(len(matrix_rows) * 35 + 38, 800),
            )

            st.divider()

            # Per-platform summary
            st.markdown("### Platform Coverage Summary")
            summary_rows = []
            for pid in PLAT_IDS:
                mapped_count = plat_coverage.get(pid, 0)
                pname = PLAT_NAMES[pid]
                total_cols = sum(
                    1 for c in _ap_source if c.get("platform_id") == pid
                )
                approved_cols = sum(
                    1 for m in _ap_approved
                    if _ap_col_lookup.get(m.get("column_id"), {}).get("platform_id") == pid
                )
                summary_rows.append({
                    "Platform": pname,
                    "Canonical Fields Covered": mapped_count,
                    "Source Columns": total_cols,
                    "Approved Columns": approved_cols,
                    "Column Coverage": f"{approved_cols / total_cols * 100:.0f}%" if total_cols else "0%",
                })
            st.dataframe(
                pd.DataFrame(summary_rows),
                use_container_width=True,
                hide_index=True,
            )

            # Download
            csv_data = matrix_df[show_cols].to_csv(index=False)
            st.download_button(
                "Download Mapping Matrix (CSV)",
                csv_data,
                "approved_mappings_matrix.csv",
                "text/csv",
                key="dl_matrix",
            )


# =====================================================================
# PAGE 3: MASTER FILE MANAGEMENT
# =====================================================================
with tab_master:
    st.markdown("## Master File Management")
    st.caption(
        "Manage standardized master entries that serve as the canonical reference "
        "for data mapping across all platforms."
    )

    canonical_raw = load_canonical()
    approved = load_approved_mappings()
    source_cols = load_source_columns()

    mf_col_lookup = {c["column_id"]: c for c in source_cols}

    mapped_column_ids = {m.get("column_id") for m in approved}
    total_source_cols = len(source_cols)
    mapped_count = sum(1 for c in source_cols if c["column_id"] in mapped_column_ids)
    unmapped_count = total_source_cols - mapped_count
    coverage = (mapped_count / total_source_cols * 100) if total_source_cols else 0

    ms1, ms2, ms3, ms4 = st.columns(4)
    ms1.metric("Active Master Entries", len(canonical_raw))
    ms2.metric("Total Mappings", len(approved))
    ms3.metric("Unmapped Headers", unmapped_count)
    ms4.metric("Coverage", f"{coverage:.1f}%")

    st.divider()

    mf1, mf2 = st.columns([2, 1])
    with mf1:
        mf_search = st.text_input("Search master entries...", key="mf_search")
    with mf2:
        domains = sorted(set(c.get("domain_category", "") for c in canonical_raw if c.get("domain_category")))
        mf_domain = st.selectbox("Domain", ["All"] + domains, key="mf_domain")

    canon_display = canonical_raw
    if mf_search:
        q = mf_search.lower()
        canon_display = [
            c for c in canon_display
            if q in c.get("canonical_name", "").lower()
            or q in (c.get("business_definition") or "").lower()
        ]
    if mf_domain != "All":
        canon_display = [c for c in canon_display if c.get("domain_category") == mf_domain]

    canon_coverage = {}
    for m in approved:
        cid = m.get("canonical_id")
        col = mf_col_lookup.get(m.get("column_id"))
        if col:
            canon_coverage.setdefault(cid, {}).setdefault(col["platform_id"], []).append(col["column_name"])

    # -- Action buttons --
    btn1, btn2, btn3 = st.columns([6, 1, 1])
    with btn2:
        show_create = st.button("Create Master Entry", key="show_create_btn")
    with btn3:
        show_report = st.button("Generate Report", key="gen_report_btn")

    if show_report:
        report_data = []
        for c in canonical_raw:
            plat_cov = canon_coverage.get(c["canonical_id"], {})
            report_data.append({
                "Canonical Name": c.get("canonical_name", ""),
                "Data Type": c.get("data_type", ""),
                "Domain": c.get("domain_category", ""),
                "Definition": c.get("business_definition", ""),
                "Platforms Covered": len(plat_cov),
                "Total Mappings": sum(len(v) for v in plat_cov.values()),
            })
        st.download_button(
            "Download CSV Report",
            pd.DataFrame(report_data).to_csv(index=False),
            "master_file_report.csv",
            "text/csv",
        )

    if show_create:
        with st.container():
            st.markdown("#### New Master Entry")
            cc1, cc2 = st.columns(2)
            with cc1:
                cn_name = st.text_input("Standardized Name", key="cn_name", placeholder="e.g. fund_identifier")
                cn_type = st.selectbox("Data Type", ["string", "integer", "decimal", "date", "timestamp", "boolean"], key="cn_type")
            with cc2:
                cn_domain = st.selectbox("Domain", ["Financial", "Identity", "Temporal", "Operational", "Geographic", "Reference", "Organizational"], key="cn_domain")
                cn_def = st.text_input("Business Definition", key="cn_def", placeholder="Unique identifier for a fund entity")

            if cn_name and canonical_raw:
                similar = find_similar_canonicals(cn_name, canonical_raw, threshold=0.6)
                if similar:
                    st.warning("Similar existing entries: " + ", ".join(f"**{s[0]}** ({s[1]:.0%})" for s in similar[:5]))

            if st.button("Create Entry", type="primary", disabled=not cn_name, key="cn_create"):
                cid = uuid.uuid4().hex[:12]
                run_stmt(
                    f"INSERT INTO {T_CANONICAL} "
                    f"(canonical_id, canonical_name, data_type, business_definition, "
                    f"domain_category, is_active, created_by, created_at, updated_at) "
                    f"VALUES ({_esc(cid)}, {_esc(cn_name)}, {_esc(cn_type)}, "
                    f"{_esc(cn_def)}, {_esc(cn_domain)}, true, "
                    f"{_esc(current_user)}, current_timestamp(), current_timestamp())"
                )
                log_audit("canonical", cid, "created", current_user, {"name": cn_name})
                invalidate()
                st.toast(f"Created: {cn_name}")
                st.rerun()

    if not canon_display:
        st.info("No master entries found. Create one or wait for batch proposals to be approved.")
    else:
        master_rows = []
        for c in canon_display:
            cid = c["canonical_id"]
            plat_cov = canon_coverage.get(cid, {})
            mapped_headers = sum(len(v) for v in plat_cov.values())
            coverage_text = ", ".join(
                f"{PLAT_NAMES[pid]}({len(cols)})" for pid in PLAT_IDS if pid in plat_cov
            ) if plat_cov else "--"

            master_rows.append({
                "canonical_id": cid,
                "Standardized Name": c.get("canonical_name", ""),
                "Business Definition": (c.get("business_definition") or "")[:100],
                "Platform Coverage": coverage_text,
                "Mapped Headers": mapped_headers,
                "Domain Category": c.get("domain_category", ""),
                "Data Type": c.get("data_type", ""),
                "Status": "Active",
            })

        master_df = pd.DataFrame(master_rows)

        show_master_cols = [c for c in ["Standardized Name", "Business Definition", "Platform Coverage",
                                         "Mapped Headers", "Domain Category", "Data Type", "Status"]
                           if c in master_df.columns]

        master_event = st.dataframe(
            master_df[show_master_cols], use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="single-row",
        )
        master_sel = master_event.selection.rows if master_event.selection else []

        # -- Drill-down --
        if master_sel:
            sel_canon_id = master_df.iloc[master_sel[0]]["canonical_id"]
            sel_canon = next((c for c in canonical_raw if c["canonical_id"] == sel_canon_id), None)

            if sel_canon:
                st.divider()

                plat_cov = canon_coverage.get(sel_canon_id, {})
                total_plats = len(PLATFORMS)
                covered_plats = len(plat_cov)
                total_mapped = sum(len(v) for v in plat_cov.values())
                linked_count = sum(
                    1 for m in approved if m.get("canonical_id") == sel_canon_id
                )
                all_proposals = load_proposals()
                pending_count = sum(
                    1 for p in all_proposals
                    if p.get("suggested_canonical_id") == sel_canon_id
                    and p.get("status") == "pending_review"
                )

                header_badges = " ".join(
                    platform_badge_html(pid, len(cols))
                    for pid in PLAT_IDS if pid in plat_cov
                )
                st.markdown(
                    f'<h3 style="margin-bottom:0;">{sel_canon["canonical_name"]}'
                    f'</h3>'
                    f'<div style="margin-top:4px;">{header_badges}</div>'
                    if header_badges else
                    f'<h3>{sel_canon["canonical_name"]}</h3>',
                    unsafe_allow_html=True,
                )

                if sel_canon.get("business_definition"):
                    st.caption(sel_canon["business_definition"])

                dc1, dc2, dc3, dc4 = st.columns(4)
                dc1.metric("Platform Coverage", f"{covered_plats}/{total_plats}")
                dc2.metric("Total Mappings", total_mapped)
                dc3.metric("Data Type", sel_canon.get("data_type", "N/A"))
                status_parts = []
                if linked_count:
                    status_parts.append(f"{linked_count} Linked")
                if pending_count:
                    status_parts.append(f"{pending_count} Pending")
                dc4.metric("Status", ", ".join(status_parts) if status_parts else "No mappings")

                if plat_cov:
                    plat_tabs = st.tabs([
                        f"{PLAT_NAMES.get(pid, pid)} ({len(cols)})"
                        for pid, cols in plat_cov.items()
                    ])
                    for tab, (pid, col_names) in zip(plat_tabs, plat_cov.items()):
                        with tab:
                            st.markdown(
                                f"Showing {len(col_names)} mapping(s) for "
                                f"{platform_badge_html(pid)}",
                                unsafe_allow_html=True,
                            )
                            for m in approved:
                                if m.get("canonical_id") != sel_canon_id:
                                    continue
                                col = mf_col_lookup.get(m.get("column_id"))
                                if col and col.get("platform_id") == pid:
                                    with st.container(border=True):
                                        mc1, mc2 = st.columns([3, 1])
                                        with mc1:
                                            st.markdown(
                                                f'**`{col.get("column_name", "")}`** &rarr; '
                                                f'`{sel_canon["canonical_name"]}`'
                                            )
                                        with mc2:
                                            st.markdown(
                                                '<span class="badge badge-linked">Linked</span>',
                                                unsafe_allow_html=True,
                                            )
                                        mc3, mc4 = st.columns(2)
                                        with mc3:
                                            st.caption(f"Source Table: **{col.get('source_table', '').split('.')[-1]}**")
                                        with mc4:
                                            st.caption(f"Approved By: **{m.get('approved_by', 'N/A')}**")

                            pending_for_plat = [
                                p for p in all_proposals
                                if p.get("suggested_canonical_id") == sel_canon_id
                                and p.get("status") == "pending_review"
                                and mf_col_lookup.get(p.get("column_id"), {}).get("platform_id") == pid
                            ]
                            for p in pending_for_plat:
                                pcol = mf_col_lookup.get(p.get("column_id"), {})
                                with st.container(border=True):
                                    mc1, mc2 = st.columns([3, 1])
                                    with mc1:
                                        st.markdown(
                                            f'**`{pcol.get("column_name", "")}`** &rarr; '
                                            f'`{sel_canon["canonical_name"]}`'
                                        )
                                    with mc2:
                                        st.markdown(
                                            '<span class="badge badge-pending">Pending</span>',
                                            unsafe_allow_html=True,
                                        )
                                    mc3, mc4 = st.columns(2)
                                    with mc3:
                                        st.caption(f"Source Table: **{pcol.get('source_table', '').split('.')[-1]}**")
                                    with mc4:
                                        conf = p.get("confidence_level", "N/A").title()
                                        st.caption(f"Match Confidence: **{conf}**")
                else:
                    st.info("No platform mappings yet for this canonical field.")

    # -- Generate Gold Views --
    st.divider()
    st.markdown("### Generate Gold Views")
    st.caption(
        f"Generate SQL views in `{GOLD_CATALOG}`.`{GOLD_SCHEMA}` that rename source "
        f"columns to their canonical names."
    )

    if not approved:
        st.info("Approve mappings in the Data Mapping page to generate gold views.")
    else:
        mf_canon_lookup = {c["canonical_id"]: c for c in canonical_raw}

        tables_with_mappings: dict[str, dict] = {}
        for m in approved:
            col = mf_col_lookup.get(m.get("column_id"))
            if col and col.get("source_table"):
                key = col["source_table"]
                tables_with_mappings.setdefault(key, {
                    "platform_id": col["platform_id"],
                    "table": key,
                    "mappings": [],
                })
                canon = mf_canon_lookup.get(m.get("canonical_id"), {})
                tables_with_mappings[key]["mappings"].append({
                    "column": col["column_name"],
                    "canonical": canon.get("canonical_name", "?"),
                })

        if tables_with_mappings:
            gold_rows = []
            for tbl, info in tables_with_mappings.items():
                gold_rows.append({
                    "Source Table": tbl.split(".")[-1],
                    "Platform": PLAT_NAMES.get(info["platform_id"], info["platform_id"]),
                    "Mapped Columns": len(info["mappings"]),
                    "Column Renames": ", ".join(f"{m['column']} -> {m['canonical']}" for m in info["mappings"][:3])
                                     + ("..." if len(info["mappings"]) > 3 else ""),
                })
            st.dataframe(pd.DataFrame(gold_rows), use_container_width=True, hide_index=True)

            gc1, gc2, gc3 = st.columns([1, 1, 5])
            with gc1:
                if st.button("Preview SQL", key="preview_gold"):
                    for tbl, info in tables_with_mappings.items():
                        sql = generate_gold_view_sql(tbl, approved, source_cols, canonical_raw, info["platform_id"])
                        if sql:
                            st.code(sql, language="sql")
            with gc2:
                if st.button("Execute Gold Views", type="primary", key="exec_gold"):
                    created = 0
                    for tbl, info in tables_with_mappings.items():
                        sql = generate_gold_view_sql(tbl, approved, source_cols, canonical_raw, info["platform_id"])
                        if sql:
                            try:
                                run_stmt(sql)
                                created += 1
                            except Exception as exc:
                                st.error(f"Failed to create view for {tbl}: {exc}")
                    if created:
                        log_audit("gold_views", "batch", "created", current_user, {"count": created})
                        st.toast(f"Created {created} gold view(s).")

    # -- Standardization Rules --
    st.divider()
    with st.expander("Standardization Rules"):
        st.caption("Abbreviation rules used by the batch agent during column standardization.")
        rules = load_rules()
        if rules:
            rules_df = pd.DataFrame(rules)
            display_cols = [c for c in ["rule_id", "rule_type", "pattern", "replacement", "description", "is_active"]
                           if c in rules_df.columns]
            st.dataframe(rules_df[display_cols], use_container_width=True, hide_index=True, height=300)
        else:
            st.info("No rules configured yet.")

        st.markdown("**Add a rule**")
        rc1, rc2 = st.columns(2)
        with rc1:
            new_pattern = st.text_input("Pattern (abbreviation)", key="rule_pattern")
            new_replacement = st.text_input("Replacement", key="rule_repl")
        with rc2:
            new_rule_desc = st.text_input("Description", key="rule_desc")
        if st.button("Add Rule", type="primary", disabled=not (new_pattern and new_replacement), key="add_rule"):
            rid = uuid.uuid4().hex[:12]
            run_stmt(
                f"INSERT INTO {T_RULES} "
                f"(rule_id, rule_type, pattern, replacement, description, is_active) "
                f"VALUES ({_esc(rid)}, 'abbreviation', {_esc(new_pattern)}, "
                f"{_esc(new_replacement)}, {_esc(new_rule_desc)}, true)"
            )
            log_audit("rule", rid, "created", current_user,
                      {"pattern": new_pattern, "replacement": new_replacement})
            invalidate()
            st.toast(f"Rule added: {new_pattern} -> {new_replacement}")
            st.rerun()

    # -- Audit Log --
    st.divider()
    with st.expander("Audit Log"):
        st.caption(
            "Every proposal, approval, rejection, and edit is recorded here."
        )

        audit_raw = load_audit()
        if audit_raw:
            adf = pd.DataFrame(audit_raw)
            f1, f2, f3 = st.columns(3)
            with f1:
                aud_entity = st.selectbox(
                    "Entity", ["All"] + sorted(adf["entity_type"].dropna().unique().tolist()), key="ae"
                )
            with f2:
                aud_action = st.selectbox(
                    "Action", ["All"] + sorted(adf["action"].dropna().unique().tolist()), key="aa"
                )
            with f3:
                aud_actor = st.selectbox(
                    "Actor", ["All"] + sorted(adf["actor"].dropna().unique().tolist()), key="ac"
                )

            if aud_entity != "All":
                adf = adf[adf["entity_type"] == aud_entity]
            if aud_action != "All":
                adf = adf[adf["action"] == aud_action]
            if aud_actor != "All":
                adf = adf[adf["actor"] == aud_actor]

            aud1, aud2, aud3 = st.columns(3)
            aud1.metric("Events", len(adf))
            if not adf.empty:
                aud2.metric("Unique Actors", adf["actor"].nunique())
                aud3.metric("Entity Types", adf["entity_type"].nunique())

            cols = [c for c in ["created_at", "entity_type", "entity_id", "action", "actor", "details"]
                    if c in adf.columns]
            st.dataframe(adf[cols].reset_index(drop=True), use_container_width=True, hide_index=True, height=500)
        else:
            st.info("No audit events recorded yet.")


_log("render complete")
