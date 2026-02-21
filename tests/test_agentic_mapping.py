import json

import pytest

from column_mapping.agentic_mapping import (
    _extract_first_json_object,
    build_standardization_prompt,
    safe_int_confidence,
    source_system_from_table,
)


def test_source_system_from_table() -> None:
    assert source_system_from_table(table_name="silver_salesforce_customers", table_prefix="silver") == "salesforce_customers"
    assert source_system_from_table(table_name="silver", table_prefix="silver") == ""
    assert source_system_from_table(table_name="foo_bar", table_prefix="silver") == "foo_bar"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 0),
        ("", 0),
        (0, 0),
        (0.95, 95),
        ("0.7", 70),
        (95, 95),
        ("98", 98),
        ("not-a-number", 0),
    ],
)
def test_safe_int_confidence(value, expected: int) -> None:
    assert safe_int_confidence(value) == expected


def test_extract_first_json_object() -> None:
    text = "some preface\n{ \"a\": 1 }\ntrailing"
    assert json.loads(_extract_first_json_object(text)) == {"a": 1}

    with pytest.raises(ValueError):
        _extract_first_json_object("no json here")


def test_build_standardization_prompt_contains_key_sections() -> None:
    prompt = build_standardization_prompt(
        source_system="salesforce",
        column_name="AcctID",
        rules={
            "naming_conventions": [{"description": "Use snake_case"}],
            "abbreviations": {"acct": {"value": "account", "description": "Expand acct"}},
        },
        similar_mappings=[
            {"platform_header": "Acct-ID", "standardized_header": "account_id", "domain": "Reference", "data_type": "string"}
        ],
    )

    assert "SOURCE SYSTEM: salesforce" in prompt
    assert 'COLUMN NAME: "AcctID"' in prompt
    assert "RULES:" in prompt
    assert "ABBREVIATIONS:" in prompt
    assert "SIMILAR PAST MAPPINGS:" in prompt
    assert '"confidence_score": 95' in prompt

