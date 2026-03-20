"""Mapping agent that orchestrates tool calls for column standardization + matching.

Runs a fixed retrieval pipeline (deterministic standardize -> BM25 search approved
mappings -> BM25 search canonicals -> cross-platform context -> rules) then feeds
all tool results to an LLM for synthesis into a recommendation with rationale.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from column_mapping.agent_tools import (
    deterministic_standardize,
    get_abbreviation_rules,
    get_cross_platform_context,
    search_approved_mappings,
    search_canonical_fields,
)


@dataclass
class AgentResult:
    """Structured result from the mapping agent."""

    recommended_canonical_id: str | None = None
    recommended_canonical_name: str | None = None
    confidence: int = 0
    rationale: str = ""
    standardized_name: str = ""
    data_type: str = ""
    business_definition: str = ""
    domain_category: str = ""
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    tool_results: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def build_agent_prompt(
    column_name: str,
    platform_name: str,
    rule_standardized: str,
    similar_mappings: list[dict],
    canonical_candidates: list[dict],
    cross_platform: dict[str, list[dict]],
    abbreviations: dict[str, str],
) -> str:
    """Build the LLM prompt with all tool results as context."""
    lines = [
        "You are a data standardization expert at a financial services firm.",
        "",
        "TASK: Given a source column, recommend the best canonical field.",
        "Use all the context below to make your decision.",
        "",
        f'PLATFORM: {platform_name}',
        f'SOURCE COLUMN: "{column_name}"',
        f'RULE-STANDARDIZED: "{rule_standardized}"',
    ]

    if abbreviations:
        lines.append("")
        lines.append("ACTIVE ABBREVIATION RULES:")
        for pat, repl in sorted(abbreviations.items()):
            lines.append(f"  {pat} -> {repl}")

    if similar_mappings:
        lines.append("")
        lines.append("SIMILAR APPROVED MAPPINGS (from other platforms):")
        for i, m in enumerate(similar_mappings, 1):
            lines.append(
                f'  {i}. "{m.get("column_name", "")}" ({m.get("platform_id", "")}) '
                f'-> canonical: {m.get("canonical_name", "")} '
                f'[score: {m.get("_bm25_score", 0):.1f}]'
            )

    if canonical_candidates:
        lines.append("")
        lines.append("CANDIDATE CANONICAL FIELDS:")
        for i, c in enumerate(canonical_candidates, 1):
            lines.append(
                f'  {i}. {c.get("canonical_name", "")} '
                f'({c.get("data_type", "")}, {c.get("domain_category", "")}) '
                f'- "{c.get("business_definition", "")}" '
                f'[score: {c.get("_bm25_score", 0):.1f}]'
            )

    if cross_platform:
        lines.append("")
        lines.append("CROSS-PLATFORM CONTEXT FOR TOP CANDIDATES:")
        for cname, platforms in cross_platform.items():
            lines.append(f"  {cname}:")
            for p in platforms:
                lines.append(
                    f'    - {p.get("platform_id", "")}: "{p.get("column_name", "")}"'
                )

    lines.append("")
    lines.append("INSTRUCTIONS:")
    lines.append("- Pick the best canonical field from the candidates above.")
    lines.append("- If no candidate is appropriate, set recommended_canonical_id to null.")
    lines.append("- Provide a clear rationale citing the evidence.")
    lines.append("- Suggest a standardized_name, data_type, business_definition, domain_category.")
    lines.append("- List up to 3 alternative canonical matches with reasons.")
    lines.append("")
    lines.append("Output ONLY valid JSON:")
    lines.append("{")
    lines.append('  "recommended_canonical_id": "cf_xxx" or null,')
    lines.append('  "recommended_canonical_name": "field_name" or null,')
    lines.append('  "confidence": 0-100,')
    lines.append('  "rationale": "...",')
    lines.append('  "standardized_name": "snake_case_name",')
    lines.append('  "data_type": "string|integer|decimal|date|timestamp|boolean",')
    lines.append('  "business_definition": "1-2 sentence definition",')
    lines.append('  "domain_category": "Fund|Valuation|Investor|Capital|Fees|Transaction|Security|Position|Reporting|Operations",')
    lines.append('  "alternatives": [')
    lines.append('    {"canonical_id": "...", "canonical_name": "...", "confidence": 0-100, "reason": "..."}')
    lines.append("  ]")
    lines.append("}")

    return "\n".join(lines)


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from LLM output."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0 or end <= start:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    return json.loads(text[start:end])


def _call_llm(sql_fn: Callable, llm_endpoint: str, prompt: str) -> dict[str, Any]:
    """Call LLM via ai_query and parse JSON response."""
    escaped = prompt.replace("'", "''")
    rows = sql_fn(f"SELECT ai_query('{llm_endpoint}', '{escaped}') AS result")
    if not rows:
        raise RuntimeError("ai_query returned no rows")
    return _extract_json(rows[0]["result"])


def run_mapping_agent(
    column_name: str,
    platform_id: str,
    platform_name: str,
    *,
    source_columns: list[dict],
    approved_mappings: list[dict],
    canonical_fields: list[dict],
    rules: list[dict],
    sql_fn: Callable,
    llm_endpoint: str,
    top_k: int = 5,
) -> AgentResult:
    """Run the mapping agent for a single source column.

    Executes a fixed tool pipeline then synthesizes via LLM:
        1. deterministic_standardize
        2. search_approved_mappings (BM25)
        3. search_canonical_fields (BM25)
        4. get_cross_platform_context (for top-3 candidates)
        5. get_abbreviation_rules
        6. LLM synthesis

    Args:
        column_name: raw column name from the source platform.
        platform_id: source platform identifier.
        platform_name: human-readable platform name.
        source_columns: all source columns (column_id, platform_id, column_name, ...).
        approved_mappings: all approved mappings (column_id -> canonical_id).
        canonical_fields: all canonical field definitions.
        rules: all standardization rules.
        sql_fn: callable that executes SQL and returns list[dict].
        llm_endpoint: Databricks model serving endpoint name.
        top_k: number of results for BM25 searches.

    Returns:
        AgentResult with recommendation, rationale, and full tool call log.
    """
    tool_results: dict[str, Any] = {}

    abbreviations = get_abbreviation_rules(rules)
    tool_results["abbreviations"] = abbreviations

    rule_std = deterministic_standardize(column_name, abbreviations)
    tool_results["rule_standardized"] = rule_std

    similar = search_approved_mappings(
        query=f"{column_name} {rule_std}",
        source_columns=source_columns,
        approved_mappings=approved_mappings,
        canonical_fields=canonical_fields,
        top_k=top_k,
    )
    tool_results["similar_mappings"] = similar

    candidates = search_canonical_fields(
        query=f"{column_name} {rule_std}",
        canonical_fields=canonical_fields,
        top_k=top_k,
    )
    tool_results["canonical_candidates"] = candidates

    cross_platform: dict[str, list[dict]] = {}
    seen_canonical_ids = set()
    for src in [similar, candidates]:
        for item in src[:3]:
            cid = item.get("canonical_id")
            if cid and cid not in seen_canonical_ids:
                seen_canonical_ids.add(cid)
                ctx = get_cross_platform_context(cid, approved_mappings, source_columns)
                cname = item.get("canonical_name", cid)
                if ctx:
                    cross_platform[cname] = ctx
    tool_results["cross_platform"] = cross_platform

    prompt = build_agent_prompt(
        column_name=column_name,
        platform_name=platform_name,
        rule_standardized=rule_std,
        similar_mappings=similar,
        canonical_candidates=candidates,
        cross_platform=cross_platform,
        abbreviations=abbreviations,
    )
    tool_results["prompt"] = prompt

    try:
        llm_output = _call_llm(sql_fn, llm_endpoint, prompt)
        tool_results["llm_raw"] = llm_output
    except Exception as exc:
        return AgentResult(
            standardized_name=rule_std,
            error=str(exc),
            tool_results=tool_results,
        )

    alts = llm_output.get("alternatives", [])
    if not isinstance(alts, list):
        alts = []

    return AgentResult(
        recommended_canonical_id=llm_output.get("recommended_canonical_id"),
        recommended_canonical_name=llm_output.get("recommended_canonical_name"),
        confidence=int(llm_output.get("confidence", 0)),
        rationale=llm_output.get("rationale", ""),
        standardized_name=llm_output.get("standardized_name", rule_std),
        data_type=llm_output.get("data_type", ""),
        business_definition=llm_output.get("business_definition", ""),
        domain_category=llm_output.get("domain_category", ""),
        alternatives=alts,
        tool_results=tool_results,
    )
