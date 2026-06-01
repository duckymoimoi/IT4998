#!/bin/bash
# ============================================================
# Entrypoint: import data into Elasticsearch, then run command
# ============================================================
set -e

ES_HOST="${ES_HOST:-http://elasticsearch:9200}"
ES_INDEX="${ES_INDEX:-topcv_jobs}"
CSV_FILE="${CSV_FILE:-data/topcv_balanced_1300.csv}"

echo "=========================================="
echo "  Do An - GPU Pipeline Entrypoint"
echo "=========================================="
echo "  ES_HOST:  $ES_HOST"
echo "  ES_INDEX: $ES_INDEX"
echo "  CSV_FILE: $CSV_FILE"

echo ""
echo "[1/3] Waiting for Elasticsearch..."
for i in $(seq 1 30); do
    if curl -s "$ES_HOST" > /dev/null 2>&1; then
        echo "  ES is ready."
        break
    fi
    echo "  Waiting... ($i/30)"
    sleep 2
done

echo ""
echo "[2/3] Checking index status..."
DOC_COUNT=$(curl -s "$ES_HOST/$ES_INDEX/_count" 2>/dev/null | python -c "import sys,json; print(json.load(sys.stdin).get('count',0))" 2>/dev/null || echo "0")

if [ "$DOC_COUNT" -gt "0" ] 2>/dev/null; then
    echo "  Index '$ES_INDEX' already has $DOC_COUNT docs. Skipping import."
else
    echo "  Index empty or not found. Importing data..."
    if [ -f "$CSV_FILE" ]; then
        cd /app
        python src/import_to_elastic.py --csv "$CSV_FILE" --es-host "$ES_HOST" --index "$ES_INDEX"
        echo "  Import completed."
    else
        echo "  WARNING: CSV file not found: $CSV_FILE"
        echo "  Available CSV files in data/:"
        ls -la data/*.csv 2>/dev/null || echo "  (none)"
    fi
fi

echo ""
echo "[3/3] Running main command: $@"
echo "=========================================="
exec "$@"
