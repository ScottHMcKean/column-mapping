from column_mapping.config import compute_effective_config


def test_compute_effective_config_defaults_and_fqns() -> None:
    cfg = compute_effective_config(
        config={
            "databricks": {"catalog": "c1", "schema": "s1", "table_prefix": "silver"},
            "tables": {
                "rules_table_name": "rules_tbl",
                "mappings_table_name": "mappings_tbl",
            },
            "vector_search": {"index_name": "my_index"},
            "llm": {"endpoint": "llm-x"},
            "mapping": {"top_k": 7},
        }
    )

    assert cfg.catalog == "c1"
    assert cfg.schema == "s1"
    assert cfg.table_prefix == "silver"
    assert cfg.rules_table == "c1.s1.rules_tbl"
    assert cfg.mappings_table == "c1.s1.mappings_tbl"
    assert cfg.vs_index_full_name == "c1.s1.my_index"
    assert cfg.llm_endpoint == "llm-x"
    assert cfg.top_k == 7


def test_compute_effective_config_overrides() -> None:
    cfg = compute_effective_config(
        config={"databricks": {"catalog": "c1", "schema": "s1"}},
        catalog="c2",
        schema="s2",
        rules_table="x.y.rules",
        mappings_table="x.y.mappings",
        vs_index_name_or_full="x.y.idx",
        vs_endpoint_name="ep1",
        embedding_model_endpoint="emb1",
        llm_endpoint="llm2",
        top_k=3,
    )

    assert cfg.catalog == "c2"
    assert cfg.schema == "s2"
    assert cfg.rules_table == "x.y.rules"
    assert cfg.mappings_table == "x.y.mappings"
    assert cfg.vs_index_full_name == "x.y.idx"
    assert cfg.vs_endpoint_name == "ep1"
    assert cfg.embedding_model_endpoint == "emb1"
    assert cfg.llm_endpoint == "llm2"
    assert cfg.top_k == 3

