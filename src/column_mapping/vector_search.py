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


def validate_endpoint(*, vsc, endpoint_name: str) -> str:
    """Validate that the named endpoint exists and is ONLINE.

    Raises immediately if the name is blank or the endpoint is not found / not ONLINE.
    This function never creates an endpoint.
    """
    if not endpoint_name or not endpoint_name.strip():
        raise ValueError(
            "Vector Search endpoint name is blank. "
            "Set 'vector_search.endpoint_name' in config.yaml or pass the "
            "'vs_endpoint_name' parameter."
        )

    endpoint_name = endpoint_name.strip()

    try:
        ep = vsc.get_endpoint(endpoint_name)
    except Exception as exc:
        raise ValueError(
            f"Vector Search endpoint '{endpoint_name}' was not found. "
            f"Create it manually in the Databricks workspace before running this job."
        ) from exc

    state = _endpoint_state(ep)
    if state != "ONLINE":
        raise ValueError(
            f"Vector Search endpoint '{endpoint_name}' exists but is not ONLINE "
            f"(state={state}). Wait for it to finish provisioning or use a different endpoint."
        )

    return endpoint_name


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
