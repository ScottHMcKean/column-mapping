from __future__ import annotations

import csv
import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame, SparkSession

from column_mapping.workspace_paths import RepoPaths


def _read_csv(spark: "SparkSession", path: str, header: bool = True) -> "DataFrame":
    """Read a CSV into a Spark DataFrame.

    Serverless compute forbids spark.read.csv() on workspace paths,
    so we read via Python I/O and createDataFrame instead.
    """
    import pandas as pd

    clean_path = path.removeprefix("file:")

    pdf = pd.read_csv(clean_path)
    return spark.createDataFrame(pdf)


def ensure_demo_tables(
    *,
    spark: "SparkSession",
    paths: RepoPaths,
    catalog: str,
    schema: str,
    table_prefix: str,
    rules_table: str,
    mappings_table: str,
    canonical_columns_table: str,
) -> dict[str, str]:
    """Create/overwrite demo Delta tables from CSVs committed in the repo."""
    from pyspark.sql.functions import col, current_timestamp, lit, to_timestamp

    spark.sql(f"USE CATALOG {catalog}")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

    salesforce_table = f"{catalog}.{schema}.{table_prefix}_salesforce_customers"
    shopify_table = f"{catalog}.{schema}.{table_prefix}_shopify_orders"

    sf_df = _read_csv(spark, paths.data_file("salesforce_customers.csv"))
    sh_df = _read_csv(spark, paths.data_file("shopify_orders.csv"))

    sf_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(salesforce_table)
    sh_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(shopify_table)

    # Rules table
    rules_df = _read_csv(spark, paths.data_file("column_rules.csv"))
    rules_df = rules_df.withColumn("is_active", col("is_active").cast("boolean"))
    rules_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(rules_table)

    # Canonical columns table
    canonical_df = _read_csv(spark, paths.data_file("canonical_columns.csv"))
    canonical_df = (
        canonical_df.withColumn("canonical_name", col("canonical_name").cast("string"))
        .withColumn("business_definition", col("business_definition").cast("string"))
        .withColumn("domain", col("domain").cast("string"))
        .withColumn("expected_data_type", col("expected_data_type").cast("string"))
        .withColumn("created_by", col("created_by").cast("string"))
        .withColumn("created_at", current_timestamp())
    )
    canonical_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(canonical_columns_table)

    # Mappings table (Type 2 SCD schema)
    mappings_df = _read_csv(spark, paths.data_file("column_mappings.csv"))

    mappings_df = (
        mappings_df.withColumn("mapping_id", col("mapping_id").cast("string"))
        .withColumn("version", col("version").cast("int"))
        .withColumn("confidence_score", col("confidence_score").cast("int"))
        .withColumn("valid_from", to_timestamp(col("valid_from")))
        .withColumn("valid_to", to_timestamp(col("valid_to")))
        .withColumn("changed_by", col("changed_by").cast("string"))
        .withColumn("change_reason", col("change_reason").cast("string"))
        .withColumn("transformation_notes", col("transformation_notes").cast("string"))
        .withColumn("domain", col("domain").cast("string"))
        .withColumn("data_type", col("data_type").cast("string"))
        .withColumn("platform_header", col("platform_header").cast("string"))
        .withColumn("standardized_header", col("standardized_header").cast("string"))
        .withColumn("source_system", col("source_system").cast("string"))
        .withColumn("approval_status", col("approval_status").cast("string"))
    )

    for optional_col, dtype in [
        ("transformation_notes", "string"),
        ("valid_to", "timestamp"),
        ("changed_by", "string"),
        ("change_reason", "string"),
    ]:
        if optional_col not in mappings_df.columns:
            mappings_df = mappings_df.withColumn(optional_col, lit(None).cast(dtype))

    mappings_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(mappings_table)

    return {
        "salesforce_table": salesforce_table,
        "shopify_table": shopify_table,
        "rules_table": rules_table,
        "canonical_columns_table": canonical_columns_table,
        "mappings_table": mappings_table,
    }
