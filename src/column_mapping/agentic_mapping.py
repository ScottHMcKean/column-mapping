from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import SparkSession

from column_mapping.rules import format_rules_for_prompt, load_active_rules


def similarity_search_mappings(
    *,
    index,
    column_name: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    results = index.similarity_search(
        query_text=column_name,
        columns=[
            "platform_header",
            "standardized_header",
            "domain",
            "data_type",
            "transformation_notes",
            "confidence_score",
        ],
        num_results=top_k,
    )

    mappings = []
    for row in results["result"]["data_array"]:
        mappings.append(
            {
                "platform_header": row[0],
                "standardized_header": row[1],
                "domain": row[2],
                "data_type": row[3],
                "transformation_notes": row[4],
                "confidence_score": row[5],
            }
        )
    return mappings


def build_standardization_prompt(
    *,
    source_system: str,
    column_name: str,
    rules: Dict[str, Any],
    similar_mappings: List[Dict[str, Any]],
) -> str:
    prompt_lines = []
    prompt_lines.append("You are a data standardization expert.")
    prompt_lines.append("Standardize the following column name.")
    prompt_lines.append("")
    prompt_lines.append(f"SOURCE SYSTEM: {source_system}")
    prompt_lines.append(f'COLUMN NAME: "{column_name}"')
    prompt_lines.append("")
    prompt_lines.append(format_rules_for_prompt(rules))
    prompt_lines.append("")
    prompt_lines.append("SIMILAR PAST MAPPINGS:")
    for i, m in enumerate(similar_mappings, 1):
        prompt_lines.append(
            f"{i}. '{m['platform_header']}' -> '{m['standardized_header']}' "
            f"(Domain: {m['domain']}, Type: {m['data_type']})"
        )
    prompt_lines.append("")
    prompt_lines.append("Output ONLY valid JSON with this exact structure:")
    prompt_lines.append("{")
    prompt_lines.append('  "standardized_header": "column_name_in_snake_case",')
    prompt_lines.append('  "domain": "Customer Data|Financial|Operational|Reference|Location|Organizational",')
    prompt_lines.append('  "data_type": "string|decimal|date|timestamp|integer|boolean",')
    prompt_lines.append('  "confidence_score": 95,')
    prompt_lines.append('  "reasoning": "brief explanation"')
    prompt_lines.append("}")
    return "\n".join(prompt_lines)


def _extract_first_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0 or end <= start:
        raise ValueError("No JSON object found in model response")
    return text[start:end]


def call_ai_query_json(
    *,
    spark: "SparkSession",
    llm_endpoint: str,
    prompt: str,
) -> Dict[str, Any]:
    # Escape single quotes for embedding inside a SQL literal.
    prompt_sql = prompt.replace("'", "''")
    response = spark.sql(
        f"""
        SELECT ai_query(
            '{llm_endpoint}',
            '{prompt_sql}'
        ) AS result
        """
    ).collect()[0]["result"]

    json_str = _extract_first_json_object(response)
    return json.loads(json_str)


def standardize_column_agentic(
    *,
    spark: "SparkSession",
    index,
    rules_table: str,
    llm_endpoint: str,
    column_name: str,
    source_system: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Existing logic, made explicit as a step-by-step "agentic" workflow.

    Steps:
    - Retrieve similar mappings from Vector Search
    - Load active rules from rules table
    - Compose a prompt
    - Ask the LLM (ai_query) for a JSON answer
    """
    similar = similarity_search_mappings(index=index, column_name=column_name, top_k=top_k)
    rules = load_active_rules(spark, rules_table)
    prompt = build_standardization_prompt(
        source_system=source_system, column_name=column_name, rules=rules, similar_mappings=similar
    )

    try:
        result = call_ai_query_json(spark=spark, llm_endpoint=llm_endpoint, prompt=prompt)
        result["platform_header"] = column_name
        result["source_system"] = source_system
        return result
    except Exception as e:
        return {
            "platform_header": column_name,
            "source_system": source_system,
            "standardized_header": None,
            "domain": None,
            "data_type": None,
            "confidence_score": 0,
            "reasoning": f"Error: {e}",
        }


def discover_managed_tables(
    *, spark: "SparkSession", catalog: str, schema: str, table_prefix: str
) -> List[str]:
    discovery_query = f"""
        SELECT table_name
        FROM {catalog}.information_schema.tables
        WHERE table_catalog = '{catalog}'
          AND table_schema = '{schema}'
          AND table_name LIKE '{table_prefix}%'
          AND table_type IN ('MANAGED', 'EXTERNAL')
        ORDER BY table_name
    """
    rows = spark.sql(discovery_query).collect()
    return [r.table_name for r in rows]


def source_system_from_table(*, table_name: str, table_prefix: str) -> str:
    # Example: silver_salesforce_customers -> salesforce_customers
    prefix = f"{table_prefix}_"
    if table_name.startswith(prefix):
        return table_name[len(prefix) :]
    if table_name.startswith(table_prefix):
        return table_name[len(table_prefix) :].lstrip("_")
    return table_name


def safe_int_confidence(value: Any) -> int:
    if value is None:
        return 0
    try:
        # Support both 0-1 floats and 0-100 ints
        f = float(value)
        return int(round(f * 100)) if f <= 1.0 else int(round(f))
    except Exception:
        return 0

