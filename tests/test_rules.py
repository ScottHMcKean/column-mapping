from column_mapping.rules import format_rules_for_prompt


def test_format_rules_for_prompt() -> None:
    text = format_rules_for_prompt(
        {
            "naming_conventions": [{"description": "Use snake_case"}],
            "abbreviations": {"acct": {"value": "account", "description": "Expand acct"}},
        }
    )
    assert "RULES:" in text
    assert "- Use snake_case" in text
    assert "ABBREVIATIONS:" in text
    assert "'acct'" in text
    assert "account" in text

