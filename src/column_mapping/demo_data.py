from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame, SparkSession

from column_mapping.workspace_paths import RepoPaths


def _read_csv(spark: "SparkSession", path: str, header: bool = True) -> "DataFrame":
    return (
        spark.read.option("header", str(header).lower())
        .option("inferSchema", "true")
        .csv(path)
    )


def ensure_demo_tables(
    *,
    spark: "SparkSession",
    paths: RepoPaths,
    catalog: str,
    schema: str,
    table_prefix: str,
    rules_table: str,
    mappings_table: str,
) -> dict[str, str]:
    """Create/overwrite demo Delta tables from CSVs committed in the repo."""
    from pyspark.sql.functions import col, lit, to_timestamp
    # Avoid requiring CREATE CATALOG privileges. Expect the catalog to already exist.
    spark.sql(f"USE CATALOG {catalog}")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

    salesforce_table = f"{catalog}.{schema}.{table_prefix}_salesforce_customers"
    shopify_table = f"{catalog}.{schema}.{table_prefix}_shopify_orders"

    sf_df = _read_csv(spark, paths.data_file("salesforce_customers.csv"))
    sh_df = _read_csv(spark, paths.data_file("shopify_orders.csv"))

    # Persist sample source tables
    sf_df.write.mode("overwrite").saveAsTable(salesforce_table)
    sh_df.write.mode("overwrite").saveAsTable(shopify_table)

    # Rules table
    rules_df = _read_csv(spark, paths.data_file("column_rules.csv"))
    rules_df = rules_df.withColumn("is_active", col("is_active").cast("boolean"))
    rules_df.write.mode("overwrite").saveAsTable(rules_table)

    # Mappings table
    mappings_df = _read_csv(spark, paths.data_file("column_mappings.csv"))

    # Normalize types for VS + downstream appends
    mappings_df = (
        mappings_df.withColumn("mapping_id", col("mapping_id").cast("string"))
        .withColumn("confidence_score", col("confidence_score").cast("int"))
        .withColumn("approved_by", col("approved_by").cast("string"))
        .withColumn("approval_status", col("approval_status").cast("string"))
        .withColumn("approved_at", to_timestamp(col("approved_at")))
        .withColumn("created_at", to_timestamp(col("created_at")))
        .withColumn("transformation_notes", col("transformation_notes").cast("string"))
        .withColumn("domain", col("domain").cast("string"))
        .withColumn("data_type", col("data_type").cast("string"))
        .withColumn("platform_header", col("platform_header").cast("string"))
        .withColumn("standardized_header", col("standardized_header").cast("string"))
        .withColumn("source_system", col("source_system").cast("string"))
    )

    # Ensure missing optional columns exist (future-safe for appends)
    for optional_col, dtype in [
        ("transformation_notes", "string"),
        ("approved_by", "string"),
        ("approved_at", "timestamp"),
        ("created_at", "timestamp"),
    ]:
        if optional_col not in mappings_df.columns:
            mappings_df = mappings_df.withColumn(optional_col, lit(None).cast(dtype))

    mappings_df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(mappings_table)

    return {
        "salesforce_table": salesforce_table,
        "shopify_table": shopify_table,
        "rules_table": rules_table,
        "mappings_table": mappings_table,
    }

