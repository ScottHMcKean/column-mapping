"""Tests for the mapping agent orchestrator (prompt building, result parsing)."""

import json

import pytest
from column_mapping.agent_tools import prepare_search_context
from column_mapping.mapping_agent import (
    AgentResult,
    _extract_json,
    build_agent_prompt,
    build_prompt_for_column,
    parse_llm_response,
    run_mapping_agent,
)


CANONICAL_FIELDS = [
    {"canonical_id": "cf_001", "canonical_name": "fund_identifier", "data_type": "STRING",
     "business_definition": "Unique identifier for a fund entity", "domain_category": "Fund"},
    {"canonical_id": "cf_002", "canonical_name": "fund_name", "data_type": "STRING",
     "business_definition": "Legal name of the fund", "domain_category": "Fund"},
]

SOURCE_COLUMNS = [
    {"column_id": "sc_001", "platform_id": "alpha_ledger", "column_name": "FND_ID",
     "source_table": "column_mapping.alpha_ledger.positions", "data_type": "STRING"},
]

APPROVED_MAPPINGS = [
    {"mapping_id": "am_001", "column_id": "sc_001", "canonical_id": "cf_001"},
]

RULES = [
    {"rule_id": "r1", "rule_type": "abbreviation", "pattern": "fnd", "replacement": "fund",
     "is_active": True},
]


class TestExtractJson:
    def test_clean_json(self):
        raw = '{"key": "value"}'
        assert _extract_json(raw) == {"key": "value"}

    def test_json_with_preamble(self):
        raw = 'Here is my answer: {"key": "value"} done.'
        assert _extract_json(raw) == {"key": "value"}

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No JSON"):
            _extract_json("no json here")

    def test_nested_json(self):
        raw = '{"outer": {"inner": 1}, "list": [1,2]}'
        result = _extract_json(raw)
        assert result["outer"]["inner"] == 1
        assert result["list"] == [1, 2]


class TestBuildAgentPrompt:
    def test_contains_column_info(self):
        prompt = build_agent_prompt(
            column_name="FND_ID",
            platform_name="Alpha Ledger",
            rule_standardized="fund_id",
            similar_mappings=[],
            canonical_candidates=[],
            cross_platform={},
            abbreviations={},
        )
        assert "FND_ID" in prompt
        assert "Alpha Ledger" in prompt
        assert "fund_id" in prompt

    def test_contains_similar_mappings(self):
        similar = [{
            "column_name": "FUND_CODE",
            "platform_id": "capital_track",
            "canonical_name": "fund_identifier",
            "_bm25_score": 3.5,
        }]
        prompt = build_agent_prompt(
            column_name="FND_ID",
            platform_name="Alpha Ledger",
            rule_standardized="fund_id",
            similar_mappings=similar,
            canonical_candidates=[],
            cross_platform={},
            abbreviations={},
        )
        assert "SIMILAR APPROVED MAPPINGS" in prompt
        assert "FUND_CODE" in prompt
        assert "capital_track" in prompt

    def test_contains_canonical_candidates(self):
        prompt = build_agent_prompt(
            column_name="FND_ID",
            platform_name="Alpha Ledger",
            rule_standardized="fund_id",
            similar_mappings=[],
            canonical_candidates=CANONICAL_FIELDS,
            cross_platform={},
            abbreviations={},
        )
        assert "CANDIDATE CANONICAL FIELDS" in prompt
        assert "fund_identifier" in prompt
        assert "Unique identifier for a fund entity" in prompt

    def test_contains_cross_platform(self):
        cross_platform = {
            "fund_identifier": [
                {"platform_id": "alpha_ledger", "column_name": "FND_ID"},
                {"platform_id": "summit_books", "column_name": "fund_id"},
            ]
        }
        prompt = build_agent_prompt(
            column_name="FundCode",
            platform_name="Capital Track",
            rule_standardized="fund_code",
            similar_mappings=[],
            canonical_candidates=[],
            cross_platform=cross_platform,
            abbreviations={},
        )
        assert "CROSS-PLATFORM CONTEXT" in prompt
        assert "alpha_ledger" in prompt
        assert "summit_books" in prompt

    def test_contains_abbreviation_rules(self):
        prompt = build_agent_prompt(
            column_name="FND_ID",
            platform_name="Alpha Ledger",
            rule_standardized="fund_id",
            similar_mappings=[],
            canonical_candidates=[],
            cross_platform={},
            abbreviations={"fnd": "fund", "acct": "account"},
        )
        assert "ABBREVIATION RULES" in prompt
        assert "fnd -> fund" in prompt

    def test_json_output_format(self):
        prompt = build_agent_prompt(
            column_name="X",
            platform_name="Y",
            rule_standardized="x",
            similar_mappings=[],
            canonical_candidates=[],
            cross_platform={},
            abbreviations={},
        )
        assert "recommended_canonical_id" in prompt
        assert "rationale" in prompt
        assert "alternatives" in prompt


class TestAgentResult:
    def test_defaults(self):
        r = AgentResult()
        assert r.recommended_canonical_id is None
        assert r.confidence == 0
        assert r.alternatives == []
        assert r.error is None

    def test_with_values(self):
        r = AgentResult(
            recommended_canonical_id="cf_001",
            recommended_canonical_name="fund_identifier",
            confidence=95,
            rationale="High BM25 score and cross-platform agreement.",
        )
        assert r.recommended_canonical_id == "cf_001"
        assert r.confidence == 95


class TestBuildPromptForColumn:
    def test_returns_prompt_string(self):
        prompt = build_prompt_for_column(
            column_name="FND_ID",
            platform_name="Alpha Ledger",
            source_columns=SOURCE_COLUMNS,
            approved_mappings=APPROVED_MAPPINGS,
            canonical_fields=CANONICAL_FIELDS,
            rules=RULES,
        )
        assert isinstance(prompt, str)
        assert "FND_ID" in prompt
        assert "Alpha Ledger" in prompt

    def test_with_prebuilt_indices(self):
        ctx = prepare_search_context(SOURCE_COLUMNS, APPROVED_MAPPINGS, CANONICAL_FIELDS)
        prompt = build_prompt_for_column(
            column_name="FND_ID",
            platform_name="Alpha Ledger",
            source_columns=SOURCE_COLUMNS,
            approved_mappings=APPROVED_MAPPINGS,
            canonical_fields=CANONICAL_FIELDS,
            rules=RULES,
            prebuilt=ctx,
        )
        assert isinstance(prompt, str)
        assert "FND_ID" in prompt
        assert "RULE-STANDARDIZED" in prompt

    def test_prebuilt_matches_fresh(self):
        ctx = prepare_search_context(SOURCE_COLUMNS, APPROVED_MAPPINGS, CANONICAL_FIELDS)
        fresh = build_prompt_for_column(
            column_name="FND_ID",
            platform_name="Alpha Ledger",
            source_columns=SOURCE_COLUMNS,
            approved_mappings=APPROVED_MAPPINGS,
            canonical_fields=CANONICAL_FIELDS,
            rules=RULES,
        )
        prebuilt = build_prompt_for_column(
            column_name="FND_ID",
            platform_name="Alpha Ledger",
            source_columns=SOURCE_COLUMNS,
            approved_mappings=APPROVED_MAPPINGS,
            canonical_fields=CANONICAL_FIELDS,
            rules=RULES,
            prebuilt=ctx,
        )
        assert fresh == prebuilt


class TestParseLlmResponse:
    def test_valid_response(self):
        raw = json.dumps({
            "recommended_canonical_id": "cf_001",
            "recommended_canonical_name": "fund_identifier",
            "confidence": 90,
            "rationale": "Good match.",
            "standardized_name": "fund_id",
            "data_type": "STRING",
            "business_definition": "Fund ID",
            "domain_category": "Fund",
            "alternatives": [],
        })
        result = parse_llm_response(raw)
        assert result.error is None
        assert result.recommended_canonical_id == "cf_001"
        assert result.confidence == 90

    def test_invalid_json(self):
        result = parse_llm_response("not json at all")
        assert result.error is not None
        assert "parse" in result.error.lower() or "json" in result.error.lower()

    def test_fallback_standardized(self):
        result = parse_llm_response("bad", fallback_standardized="my_col")
        assert result.standardized_name == "my_col"


class TestRunMappingAgent:
    def test_llm_error_returns_result_with_error(self):
        def failing_sql(query):
            if "ai_query" in query:
                raise RuntimeError("LLM unavailable")
            return []

        result = run_mapping_agent(
            column_name="FND_ID",
            platform_id="alpha_ledger",
            platform_name="Alpha Ledger",
            source_columns=SOURCE_COLUMNS,
            approved_mappings=APPROVED_MAPPINGS,
            canonical_fields=CANONICAL_FIELDS,
            rules=RULES,
            sql_fn=failing_sql,
            llm_endpoint="test-endpoint",
        )
        assert result.error is not None
        assert "LLM unavailable" in result.error
        assert result.tool_results.get("rule_standardized") == "fund_id"
        assert "prompt" in result.tool_results

    def test_successful_llm_returns_result(self):
        llm_response = json.dumps({
            "recommended_canonical_id": "cf_001",
            "recommended_canonical_name": "fund_identifier",
            "confidence": 95,
            "rationale": "FND_ID maps to fund_identifier across 2 platforms.",
            "standardized_name": "fund_identifier",
            "data_type": "STRING",
            "business_definition": "Unique fund identifier",
            "domain_category": "Fund",
            "alternatives": [
                {"canonical_id": "cf_002", "canonical_name": "fund_name", "confidence": 30,
                 "reason": "Name contains 'fund' but semantics differ"}
            ],
        })

        def mock_sql(query):
            if "ai_query" in query:
                return [{"result": llm_response}]
            return []

        result = run_mapping_agent(
            column_name="FND_ID",
            platform_id="capital_track",
            platform_name="Capital Track",
            source_columns=SOURCE_COLUMNS,
            approved_mappings=APPROVED_MAPPINGS,
            canonical_fields=CANONICAL_FIELDS,
            rules=RULES,
            sql_fn=mock_sql,
            llm_endpoint="test-endpoint",
        )
        assert result.error is None
        assert result.recommended_canonical_id == "cf_001"
        assert result.confidence == 95
        assert "fund_identifier" in result.rationale
        assert len(result.alternatives) == 1
        assert result.tool_results.get("rule_standardized") == "fund_id"

    def test_with_prebuilt_indices(self):
        llm_response = json.dumps({
            "recommended_canonical_id": "cf_001",
            "recommended_canonical_name": "fund_identifier",
            "confidence": 92,
            "rationale": "Strong match.",
            "standardized_name": "fund_id",
            "data_type": "STRING",
            "business_definition": "Unique fund identifier",
            "domain_category": "Fund",
            "alternatives": [],
        })

        def mock_sql(query):
            if "ai_query" in query:
                return [{"result": llm_response}]
            return []

        ctx = prepare_search_context(SOURCE_COLUMNS, APPROVED_MAPPINGS, CANONICAL_FIELDS)
        result = run_mapping_agent(
            column_name="FND_ID",
            platform_id="alpha_ledger",
            platform_name="Alpha Ledger",
            source_columns=SOURCE_COLUMNS,
            approved_mappings=APPROVED_MAPPINGS,
            canonical_fields=CANONICAL_FIELDS,
            rules=RULES,
            sql_fn=mock_sql,
            llm_endpoint="test-endpoint",
            prebuilt=ctx,
        )
        assert result.error is None
        assert result.recommended_canonical_id == "cf_001"
