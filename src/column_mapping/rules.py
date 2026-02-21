from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import SparkSession


def load_active_rules(spark: "SparkSession", rules_table: str) -> Dict[str, Any]:
    rules_df = spark.sql(
        f"""
        SELECT rule_type, rule_key, rule_value, rule_description, examples
        FROM {rules_table}
        WHERE is_active = TRUE
        """
    )

    rules: Dict[str, Any] = {
        "naming_conventions": [],
        "abbreviations": {},
        "domains": {},
        "data_types": {},
    }

    for row in rules_df.collect():
        if row.rule_type == "naming_convention":
            rules["naming_conventions"].append(
                {"key": row.rule_key, "value": row.rule_value, "description": row.rule_description}
            )
        elif row.rule_type == "abbreviation":
            rules["abbreviations"][row.rule_key] = {
                "value": row.rule_value,
                "description": row.rule_description,
            }
        elif row.rule_type == "domain":
            rules["domains"][row.rule_key] = {
                "value": row.rule_value,
                "description": row.rule_description,
            }
        elif row.rule_type == "data_type":
            rules["data_types"][row.rule_key] = {
                "value": row.rule_value,
                "description": row.rule_description,
            }
    return rules


def format_rules_for_prompt(rules: Dict[str, Any]) -> str:
    lines = []
    lines.append("RULES:")
    for conv in rules.get("naming_conventions", []):
        lines.append(f"- {conv['description']}")
    if rules.get("abbreviations"):
        lines.append("")
        lines.append("ABBREVIATIONS:")
        for key, val in rules["abbreviations"].items():
            lines.append(f"- '{key}' -> '{val['value']}': {val.get('description', '')}")
    return "\n".join(lines)

