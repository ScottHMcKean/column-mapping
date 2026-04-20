"""Shared tool library for the column mapping agent.

Provides BM25 keyword search over approved mappings and canonical fields,
cross-platform context lookup, deterministic standardization, and abbreviation
rule access.
"""

from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi


SQL_RESERVED = {
    "select", "from", "where", "table", "column", "index", "group", "order",
    "by", "join", "on", "insert", "update", "delete", "create", "drop",
    "alter", "set", "values", "into", "null", "not", "and", "or", "in",
    "is", "like", "between", "exists", "having", "limit", "union", "all",
    "distinct", "as", "case", "when", "then", "else", "end", "cast", "with",
}


def _tokenize(text: str) -> list[str]:
    """Split on whitespace, underscores, and hyphens. Lowercase."""
    if not text:
        return []
    return [t for t in re.split(r"[\s_\-\.\/,;:]+", text.lower()) if t]


def build_bm25_index(
    documents: list[dict[str, Any]],
    text_field: str,
) -> tuple[BM25Okapi | None, list[dict[str, Any]]]:
    """Build a BM25 index from a list of dicts.

    Args:
        documents: list of dicts, each containing at least ``text_field``.
        text_field: key whose value is tokenized for the index.

    Returns:
        (bm25_index, documents) or (None, []) if documents is empty.
    """
    if not documents:
        return None, []
    corpus = [_tokenize(doc.get(text_field, "")) for doc in documents]
    return BM25Okapi(corpus), documents


def _bm25_search(
    query: str,
    index: BM25Okapi | None,
    documents: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Search a pre-built BM25 index. Returns top-k docs with scores."""
    if index is None or not documents:
        return []
    tokens = _tokenize(query)
    if not tokens:
        return []
    scores = index.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in ranked[:top_k]:
        if score > 0:
            results.append({**documents[idx], "_bm25_score": round(float(score), 4)})
    return results


def _join_approved_docs(
    source_columns: list[dict],
    approved_mappings: list[dict],
    canonical_fields: list[dict],
) -> list[dict[str, Any]]:
    """Join source -> approved -> canonical into searchable documents."""
    canon_by_id = {c["canonical_id"]: c for c in canonical_fields}
    col_by_id = {c["column_id"]: c for c in source_columns}

    joined = []
    for m in approved_mappings:
        col = col_by_id.get(m.get("column_id"))
        if not col:
            continue
        canon = canon_by_id.get(m.get("canonical_id"), {})
        joined.append({
            "column_id": col["column_id"],
            "platform_id": col.get("platform_id", ""),
            "column_name": col.get("column_name", ""),
            "source_table": col.get("source_table", ""),
            "canonical_id": m.get("canonical_id", ""),
            "canonical_name": canon.get("canonical_name", ""),
            "business_definition": canon.get("business_definition", ""),
            "domain_category": canon.get("domain_category", ""),
            "data_type": canon.get("data_type", ""),
            "_search_text": " ".join([
                col.get("column_name", ""),
                canon.get("canonical_name", ""),
                canon.get("business_definition", ""),
            ]),
        })
    return joined


def _enrich_canonical_docs(canonical_fields: list[dict]) -> list[dict[str, Any]]:
    """Add composite search text to canonical field dicts."""
    return [
        {
            **c,
            "_search_text": " ".join([
                c.get("canonical_name", ""),
                c.get("business_definition", ""),
                c.get("domain_category", ""),
            ]),
        }
        for c in canonical_fields
    ]


def prepare_search_context(
    source_columns: list[dict],
    approved_mappings: list[dict],
    canonical_fields: list[dict],
) -> dict[str, Any]:
    """Pre-build BM25 indices for reuse across many search calls.

    Returns a dict with keys:
        approved_index, approved_docs  -- for search_approved_mappings
        canonical_index, canonical_docs -- for search_canonical_fields
    """
    approved_docs = _join_approved_docs(source_columns, approved_mappings, canonical_fields)
    approved_index, approved_docs = build_bm25_index(approved_docs, "_search_text")

    canonical_docs = _enrich_canonical_docs(canonical_fields)
    canonical_index, canonical_docs = build_bm25_index(canonical_docs, "_search_text")

    return {
        "approved_index": approved_index,
        "approved_docs": approved_docs,
        "canonical_index": canonical_index,
        "canonical_docs": canonical_docs,
    }


def search_approved_mappings(
    query: str,
    source_columns: list[dict],
    approved_mappings: list[dict],
    canonical_fields: list[dict],
    top_k: int = 5,
    *,
    prebuilt: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """BM25 search over approved column-to-canonical mappings.

    Joins source_columns -> approved_mappings -> canonical_fields and searches
    over a composite text field combining the source column name, canonical
    name, and business definition.

    Pass ``prebuilt`` (from prepare_search_context) to skip index construction.
    """
    if prebuilt:
        index = prebuilt["approved_index"]
        docs = prebuilt["approved_docs"]
    else:
        joined = _join_approved_docs(source_columns, approved_mappings, canonical_fields)
        index, docs = build_bm25_index(joined, "_search_text")

    results = _bm25_search(query, index, docs, top_k)
    for r in results:
        r.pop("_search_text", None)
    return results


def search_canonical_fields(
    query: str,
    canonical_fields: list[dict],
    top_k: int = 5,
    *,
    prebuilt: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """BM25 search over canonical fields using name + definition + domain.

    Pass ``prebuilt`` (from prepare_search_context) to skip index construction.
    """
    if prebuilt:
        index = prebuilt["canonical_index"]
        docs = prebuilt["canonical_docs"]
    else:
        enriched = _enrich_canonical_docs(canonical_fields)
        index, docs = build_bm25_index(enriched, "_search_text")

    results = _bm25_search(query, index, docs, top_k)
    for r in results:
        r.pop("_search_text", None)
    return results


def get_cross_platform_context(
    canonical_id: str,
    approved_mappings: list[dict],
    source_columns: list[dict],
) -> list[dict[str, str]]:
    """Return all platforms that map to a given canonical field.

    Each result contains: platform_id, column_name, source_table.
    """
    col_by_id = {c["column_id"]: c for c in source_columns}

    context = []
    for m in approved_mappings:
        if m.get("canonical_id") != canonical_id:
            continue
        col = col_by_id.get(m.get("column_id"), {})
        if col:
            context.append({
                "platform_id": col.get("platform_id", ""),
                "column_name": col.get("column_name", ""),
                "source_table": col.get("source_table", ""),
            })
    return context


def get_abbreviation_rules(rules: list[dict]) -> dict[str, str]:
    """Filter active abbreviation rules into {pattern: replacement}."""
    return {
        r["pattern"]: r["replacement"]
        for r in rules
        if r.get("rule_type") == "abbreviation"
        and r.get("is_active") in (True, "true", "True", "TRUE")
    }


def deterministic_standardize(header: str, abbreviations: dict | None = None) -> str:
    """Apply deterministic rules to normalize a column header.

    Steps: special patterns, parens extraction, lowercase + separators,
    strip non-alnum, abbreviation expansion, numeric prefix, max length,
    SQL reserved word avoidance.
    """
    s = header.strip()
    s = re.sub(r"#\s*of\b", "count_of", s, flags=re.IGNORECASE)
    s = re.sub(r"%", "_pct", s)

    parens = re.findall(r"\(([^)]+)\)", s)
    s = re.sub(r"\([^)]+\)", "", s)

    s = s.lower()
    s = re.sub(r"[\s\-\.\/,;:]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)

    for paren_content in parens:
        suffix = re.sub(r"[^a-z0-9]", "_", paren_content.lower()).strip("_")
        if suffix:
            s = f"{s}_{suffix}"

    s = re.sub(r"_+", "_", s).strip("_")

    if abbreviations:
        tokens = s.split("_")
        tokens = [abbreviations.get(tok, tok) for tok in tokens]
        s = "_".join(tokens)
        s = re.sub(r"_+", "_", s).strip("_")

    if s and s[0].isdigit():
        s = "n_" + s

    s = re.sub(r"_+", "_", s).strip("_")

    if len(s) > 30:
        s = s[:30].rstrip("_")

    if s in SQL_RESERVED:
        s = s + "_field"

    return s or header.lower().replace(" ", "_")[:30]
