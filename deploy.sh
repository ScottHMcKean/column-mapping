#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-default}"

echo "=== Column Mapper Deploy (target: $TARGET) ==="

echo "[1/4] Validating bundle..."
databricks bundle validate -t "$TARGET"

echo "[2/4] Deploying bundle (jobs + app)..."
databricks bundle deploy -t "$TARGET" --auto-approve

echo "[3/4] Running setup job (catalog, tables, seed data, permissions)..."
databricks bundle run column_mapping_setup -t "$TARGET"

echo "[4/4] Starting app..."
databricks bundle run column_mapper -t "$TARGET"

echo ""
echo "=== Deploy complete ==="
echo "Check app status: databricks apps get column-mapper"
echo "View app logs:    databricks apps logs column-mapper"
