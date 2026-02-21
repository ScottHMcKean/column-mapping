from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


def _deep_get(dct: Dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _read_workspace_file(dbutils, file_path: str) -> str:
    # For `file:/Workspace/...` paths, `dbutils.fs.head` works.
    return dbutils.fs.head(file_path, 1024 * 1024)


def load_repo_config(dbutils, repo_root_ws: str) -> Dict[str, Any]:
    """Load `config.yaml` from the repo root in the Databricks workspace."""
    config_path = f"file:{repo_root_ws}/config.yaml"
    raw = _read_workspace_file(dbutils, config_path)
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency `pyyaml`. Install it in the notebook with `%pip install pyyaml`."
        ) from e
    return yaml.safe_load(raw) or {}


@dataclass(frozen=True)
class EffectiveConfig:
    catalog: str
    schema: str
    table_prefix: str

    rules_table: str
    mappings_table: str

    vs_endpoint_name: Optional[str]
    vs_endpoint_type: str
    vs_index_full_name: str
    embedding_model_endpoint: str
    vs_primary_key: str
    vs_embedding_source_column: str
    vs_pipeline_type: str

    llm_endpoint: str
    top_k: int


def compute_effective_config(
    *,
    config: Dict[str, Any],
    # Optional per-run overrides (e.g., notebook widgets/job parameters)
    catalog: Optional[str] = None,
    schema: Optional[str] = None,
    table_prefix: Optional[str] = None,
    rules_table: Optional[str] = None,
    mappings_table: Optional[str] = None,
    vs_endpoint_name: Optional[str] = None,
    vs_index_name_or_full: Optional[str] = None,
    embedding_model_endpoint: Optional[str] = None,
    llm_endpoint: Optional[str] = None,
    top_k: Optional[int] = None,
) -> EffectiveConfig:
    cfg_catalog = (catalog or "").strip() or _deep_get(config, ["databricks", "catalog"], "main")
    cfg_schema = (schema or "").strip() or _deep_get(config, ["databricks", "schema"], "column_mapping_demo")
    cfg_prefix = (table_prefix or "").strip() or _deep_get(config, ["databricks", "table_prefix"], "silver")

    default_rules_name = _deep_get(config, ["tables", "rules_table_name"], "governance_standardization_rules")
    default_mappings_name = _deep_get(config, ["tables", "mappings_table_name"], "governance_standardization_mappings")
    cfg_rules_table = (rules_table or "").strip() or f"{cfg_catalog}.{cfg_schema}.{default_rules_name}"
    cfg_mappings_table = (mappings_table or "").strip() or f"{cfg_catalog}.{cfg_schema}.{default_mappings_name}"

    cfg_vs_endpoint_name = (vs_endpoint_name or "").strip() or _deep_get(config, ["vector_search", "endpoint_name"], "")
    cfg_vs_endpoint_name = cfg_vs_endpoint_name or None
    cfg_vs_endpoint_type = _deep_get(config, ["vector_search", "endpoint_type"], "STANDARD")

    index_name_default = _deep_get(config, ["vector_search", "index_name"], "governance_standardization_mappings_index")
    idx = (vs_index_name_or_full or "").strip() or index_name_default
    cfg_vs_index_full_name = idx if idx.count(".") == 2 else f"{cfg_catalog}.{cfg_schema}.{idx}"

    cfg_embedding_model_endpoint = (embedding_model_endpoint or "").strip() or _deep_get(
        config, ["vector_search", "embedding_model_endpoint"], "databricks-bge-large-en"
    )

    cfg_pk = _deep_get(config, ["vector_search", "delta_sync", "primary_key"], "mapping_id")
    cfg_embed_col = _deep_get(config, ["vector_search", "delta_sync", "embedding_source_column"], "platform_header")
    cfg_pipeline_type = _deep_get(config, ["vector_search", "delta_sync", "pipeline_type"], "TRIGGERED")

    cfg_llm_endpoint = (llm_endpoint or "").strip() or _deep_get(config, ["llm", "endpoint"], "databricks-claude-haiku-4-5")

    cfg_top_k = top_k if top_k is not None else int(_deep_get(config, ["mapping", "top_k"], 5))

    return EffectiveConfig(
        catalog=cfg_catalog,
        schema=cfg_schema,
        table_prefix=cfg_prefix,
        rules_table=cfg_rules_table,
        mappings_table=cfg_mappings_table,
        vs_endpoint_name=cfg_vs_endpoint_name,
        vs_endpoint_type=cfg_vs_endpoint_type,
        vs_index_full_name=cfg_vs_index_full_name,
        embedding_model_endpoint=cfg_embedding_model_endpoint,
        vs_primary_key=cfg_pk,
        vs_embedding_source_column=cfg_embed_col,
        vs_pipeline_type=cfg_pipeline_type,
        llm_endpoint=cfg_llm_endpoint,
        top_k=cfg_top_k,
    )

