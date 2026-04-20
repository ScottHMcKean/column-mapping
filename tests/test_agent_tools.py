"""Tests for the agent tool library (BM25 search, cross-platform context, standardization)."""

import pytest
from column_mapping.agent_tools import (
    _tokenize,
    build_bm25_index,
    deterministic_standardize,
    get_abbreviation_rules,
    get_cross_platform_context,
    prepare_search_context,
    search_approved_mappings,
    search_canonical_fields,
)


CANONICAL_FIELDS = [
    {"canonical_id": "cf_001", "canonical_name": "fund_identifier", "data_type": "STRING",
     "business_definition": "Unique identifier for a fund entity", "domain_category": "Fund"},
    {"canonical_id": "cf_002", "canonical_name": "fund_name", "data_type": "STRING",
     "business_definition": "Legal name of the fund", "domain_category": "Fund"},
    {"canonical_id": "cf_005", "canonical_name": "net_asset_value", "data_type": "DECIMAL",
     "business_definition": "Total net asset value of the fund", "domain_category": "Valuation"},
    {"canonical_id": "cf_008", "canonical_name": "investor_identifier", "data_type": "STRING",
     "business_definition": "Unique identifier for an investor", "domain_category": "Investor"},
    {"canonical_id": "cf_017", "canonical_name": "transaction_date", "data_type": "DATE",
     "business_definition": "Date the transaction occurred", "domain_category": "Transaction"},
]

SOURCE_COLUMNS = [
    {"column_id": "sc_001", "platform_id": "alpha_ledger", "column_name": "FND_ID",
     "source_table": "column_mapping.alpha_ledger.positions", "data_type": "STRING"},
    {"column_id": "sc_002", "platform_id": "alpha_ledger", "column_name": "NAV_AMT",
     "source_table": "column_mapping.alpha_ledger.positions", "data_type": "DECIMAL"},
    {"column_id": "sc_003", "platform_id": "summit_books", "column_name": "fund_id",
     "source_table": "column_mapping.summit_books.investors", "data_type": "STRING"},
]

APPROVED_MAPPINGS = [
    {"mapping_id": "am_001", "column_id": "sc_001", "canonical_id": "cf_001", "proposal_id": "mp_001"},
    {"mapping_id": "am_002", "column_id": "sc_002", "canonical_id": "cf_005", "proposal_id": "mp_002"},
    {"mapping_id": "am_003", "column_id": "sc_003", "canonical_id": "cf_001", "proposal_id": "mp_003"},
]

RULES = [
    {"rule_id": "r1", "rule_type": "abbreviation", "pattern": "fnd", "replacement": "fund", "is_active": True},
    {"rule_id": "r2", "rule_type": "abbreviation", "pattern": "acct", "replacement": "account", "is_active": True},
    {"rule_id": "r3", "rule_type": "abbreviation", "pattern": "nav", "replacement": "net_asset_value", "is_active": True},
    {"rule_id": "r4", "rule_type": "abbreviation", "pattern": "txn", "replacement": "transaction", "is_active": False},
    {"rule_id": "r5", "rule_type": "naming_convention", "pattern": "snake", "replacement": "snake_case", "is_active": True},
]


class TestTokenize:
    def test_basic(self):
        assert _tokenize("fund_identifier") == ["fund", "identifier"]

    def test_mixed_separators(self):
        assert _tokenize("Fund-Name.value") == ["fund", "name", "value"]

    def test_empty(self):
        assert _tokenize("") == []
        assert _tokenize(None) == []


class TestBuildBm25Index:
    def test_builds_index(self):
        docs = [{"text": "hello world"}, {"text": "foo bar"}]
        index, returned_docs = build_bm25_index(docs, "text")
        assert index is not None
        assert len(returned_docs) == 2

    def test_empty_docs(self):
        index, docs = build_bm25_index([], "text")
        assert index is None
        assert docs == []


class TestPrepareSearchContext:
    def test_returns_all_keys(self):
        ctx = prepare_search_context(SOURCE_COLUMNS, APPROVED_MAPPINGS, CANONICAL_FIELDS)
        assert "approved_index" in ctx
        assert "approved_docs" in ctx
        assert "canonical_index" in ctx
        assert "canonical_docs" in ctx

    def test_indices_are_usable(self):
        ctx = prepare_search_context(SOURCE_COLUMNS, APPROVED_MAPPINGS, CANONICAL_FIELDS)
        assert ctx["approved_index"] is not None
        assert len(ctx["approved_docs"]) == len(APPROVED_MAPPINGS)
        assert ctx["canonical_index"] is not None
        assert len(ctx["canonical_docs"]) == len(CANONICAL_FIELDS)

    def test_empty_data(self):
        ctx = prepare_search_context([], [], [])
        assert ctx["approved_index"] is None
        assert ctx["approved_docs"] == []
        assert ctx["canonical_index"] is None
        assert ctx["canonical_docs"] == []


class TestSearchCanonicalFields:
    def test_finds_fund_identifier(self):
        results = search_canonical_fields("fund identifier", CANONICAL_FIELDS, top_k=3)
        assert len(results) > 0
        assert results[0]["canonical_name"] == "fund_identifier"

    def test_finds_nav(self):
        results = search_canonical_fields("net asset value", CANONICAL_FIELDS, top_k=3)
        assert len(results) > 0
        assert results[0]["canonical_name"] == "net_asset_value"

    def test_finds_transaction(self):
        results = search_canonical_fields("transaction date", CANONICAL_FIELDS, top_k=3)
        assert len(results) > 0
        assert results[0]["canonical_name"] == "transaction_date"

    def test_empty_query(self):
        results = search_canonical_fields("", CANONICAL_FIELDS, top_k=3)
        assert results == []

    def test_no_match_returns_empty(self):
        results = search_canonical_fields("xyzzy gibberish", CANONICAL_FIELDS, top_k=3)
        assert isinstance(results, list)

    def test_with_prebuilt(self):
        ctx = prepare_search_context(SOURCE_COLUMNS, APPROVED_MAPPINGS, CANONICAL_FIELDS)
        results = search_canonical_fields("fund identifier", CANONICAL_FIELDS, top_k=3, prebuilt=ctx)
        assert len(results) > 0
        assert results[0]["canonical_name"] == "fund_identifier"


class TestSearchApprovedMappings:
    def test_finds_nav_distinct_from_others(self):
        results = search_approved_mappings(
            "net asset value total", SOURCE_COLUMNS, APPROVED_MAPPINGS, CANONICAL_FIELDS, top_k=3,
        )
        assert len(results) > 0
        assert results[0]["canonical_name"] == "net_asset_value"

    def test_returns_list(self):
        results = search_approved_mappings(
            "transaction date", SOURCE_COLUMNS, APPROVED_MAPPINGS, CANONICAL_FIELDS, top_k=3,
        )
        assert isinstance(results, list)

    def test_unmapped_column_excluded(self):
        cols_with_unmapped = SOURCE_COLUMNS + [
            {"column_id": "sc_999", "platform_id": "test", "column_name": "UNMAPPED",
             "source_table": "test.test.test", "data_type": "STRING"},
        ]
        results = search_approved_mappings(
            "unmapped", cols_with_unmapped, APPROVED_MAPPINGS, CANONICAL_FIELDS, top_k=5,
        )
        ids = {r["column_id"] for r in results}
        assert "sc_999" not in ids

    def test_with_prebuilt(self):
        ctx = prepare_search_context(SOURCE_COLUMNS, APPROVED_MAPPINGS, CANONICAL_FIELDS)
        results = search_approved_mappings(
            "net asset value total", SOURCE_COLUMNS, APPROVED_MAPPINGS, CANONICAL_FIELDS,
            top_k=3, prebuilt=ctx,
        )
        assert len(results) > 0
        assert results[0]["canonical_name"] == "net_asset_value"


class TestGetCrossPlatformContext:
    def test_fund_identifier_cross_platform(self):
        ctx = get_cross_platform_context("cf_001", APPROVED_MAPPINGS, SOURCE_COLUMNS)
        assert len(ctx) == 2
        platforms = {c["platform_id"] for c in ctx}
        assert platforms == {"alpha_ledger", "summit_books"}

    def test_single_platform_mapping(self):
        ctx = get_cross_platform_context("cf_005", APPROVED_MAPPINGS, SOURCE_COLUMNS)
        assert len(ctx) == 1
        assert ctx[0]["platform_id"] == "alpha_ledger"
        assert ctx[0]["column_name"] == "NAV_AMT"

    def test_unknown_canonical(self):
        ctx = get_cross_platform_context("cf_999", APPROVED_MAPPINGS, SOURCE_COLUMNS)
        assert ctx == []


class TestGetAbbreviationRules:
    def test_filters_active_abbreviations(self):
        abbrevs = get_abbreviation_rules(RULES)
        assert abbrevs == {"fnd": "fund", "acct": "account", "nav": "net_asset_value"}

    def test_excludes_inactive(self):
        abbrevs = get_abbreviation_rules(RULES)
        assert "txn" not in abbrevs

    def test_excludes_non_abbreviation(self):
        abbrevs = get_abbreviation_rules(RULES)
        assert "snake" not in abbrevs


class TestDeterministicStandardize:
    def test_snake_case(self):
        assert deterministic_standardize("Fund Name") == "fund_name"

    def test_abbreviation_expansion(self):
        assert deterministic_standardize("FND_ID", {"fnd": "fund"}) == "fund_id"

    def test_percent(self):
        result = deterministic_standardize("Occupancy Rate %")
        assert "_pct" in result

    def test_special_chars(self):
        result = deterministic_standardize("Acct-ID")
        assert result == "acct_id"

    def test_camel_case(self):
        result = deterministic_standardize("FundCode")
        assert result == "fundcode"

    def test_max_length(self):
        long_header = "this_is_a_very_long_column_header_that_exceeds_thirty_characters"
        result = deterministic_standardize(long_header)
        assert len(result) <= 35

    def test_sql_reserved(self):
        result = deterministic_standardize("SELECT")
        assert result == "select_field"

    def test_numeric_prefix(self):
        result = deterministic_standardize("1st_quarter")
        assert result.startswith("n_")
