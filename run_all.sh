#!/usr/bin/env bash
# Full pipeline. Paths come from config.py (or A3_DATA / A3_WORK / TREC_EVAL env vars).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"

# trec_eval (once)
if [ ! -x "${TREC_EVAL:-trec_eval/trec_eval}" ]; then
  git clone --depth 1 https://github.com/usnistgov/trec_eval.git && make -C trec_eval
fi

# BM25
$PY bm25.py build
$PY bm25.py search

# HNSW — two graphs from the recommended range, three efSearch values each
$PY dense.py build  --M 4 --efc 50
$PY dense.py search --M 4 --efc 50  --efs 50 100 200
$PY dense.py build  --M 8 --efc 200
$PY dense.py search --M 8 --efc 200 --efs 50 100 200
$PY evaluate.py
