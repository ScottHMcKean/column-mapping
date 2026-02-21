from __future__ import annotations

import time
from typing import Any, Dict, Optional


def _endpoint_state(endpoint: Dict[str, Any]) -> Optional[str]:
    # The VS API shape varies slightly by version; handle common patterns.
    if "state" in endpoint:
        return endpoint.get("state")
    status = endpoint.get("status") or endpoint.get("endpoint_status") or {}
    if isinstance(status, dict):
        return status.get("state")
    return None


def pick_best_active_endpoint(vsc) -> Optional[str]:
    """Return the name of an ONLINE endpoint if available, otherwise None."""
    try:
        resp = vsc.list_endpoints()
    except Exception:
        return None

    endpoints = resp.get("endpoints") or resp.get("vector_search_endpoints") or []
    online = []
    for ep in endpoints:
        name = ep.get("name")
        if not name:
            continue
        state = _endpoint_state(ep)
        if state == "ONLINE":
            online.append(name)
    # Deterministic choice so multiple notebooks/jobs pick the same endpoint.
    return sorted(online)[0] if online else None


def ensure_endpoint(
    *,
    vsc,
    preferred_name: Optional[str],
    endpoint_type: str = "STANDARD",
    create_if_missing: bool = True,
    poll_seconds: int = 5,
    timeout_seconds: int = 600,
) -> str:
    """Choose an existing ONLINE endpoint, or create/poll a named endpoint."""
    if preferred_name:
        try:
            ep = vsc.get_endpoint(preferred_name)
            state = _endpoint_state(ep)
            if state == "ONLINE":
                return preferred_name
        except Exception:
            pass

    best = pick_best_active_endpoint(vsc)
    if best:
        return best

    if not preferred_name:
        preferred_name = "column_mapping_vs_endpoint"

    if not create_if_missing:
        raise ValueError("No active Vector Search endpoint found and create_if_missing=False")

    try:
        vsc.create_endpoint(name=preferred_name, endpoint_type=endpoint_type)
    except Exception:
        # Endpoint may already exist (race); proceed to polling.
        pass

    start = time.time()
    while True:
        ep = vsc.get_endpoint(preferred_name)
        state = _endpoint_state(ep)
        if state == "ONLINE":
            return preferred_name
        if time.time() - start > timeout_seconds:
            raise TimeoutError(
                f"Vector Search endpoint '{preferred_name}' did not become ONLINE "
                f"within {timeout_seconds}s (state={state})."
            )
        time.sleep(poll_seconds)


def ensure_delta_sync_index(
    *,
    vsc,
    endpoint_name: str,
    index_full_name: str,
    source_table_full_name: str,
    primary_key: str,
    embedding_source_column: str,
    embedding_model_endpoint_name: str,
    pipeline_type: str = "TRIGGERED",
) -> Any:
    """Get an existing index or create one (idempotent)."""
    try:
        return vsc.get_index(endpoint_name=endpoint_name, index_name=index_full_name)
    except Exception:
        pass

    vsc.create_delta_sync_index(
        endpoint_name=endpoint_name,
        index_name=index_full_name,
        source_table_name=source_table_full_name,
        pipeline_type=pipeline_type,
        primary_key=primary_key,
        embedding_source_column=embedding_source_column,
        embedding_model_endpoint_name=embedding_model_endpoint_name,
    )
    return vsc.get_index(endpoint_name=endpoint_name, index_name=index_full_name)


def wait_for_index_ready(*, index, poll_seconds: int = 10, timeout_seconds: int = 900) -> Dict[str, Any]:
    start = time.time()
    while True:
        status = index.describe()
        ready = (status.get("status") or {}).get("ready")
        if ready:
            return status
        if time.time() - start > timeout_seconds:
            raise TimeoutError(f"Index did not become ready within {timeout_seconds}s")
        time.sleep(poll_seconds)

