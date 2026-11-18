#!/usr/bin/env bash
# Run the nine retrieval configurations reported in the assignment writeup.
# Requires the supplied data, Python dependencies, and external trec_eval.
# Uses A3_DATA/A3_WORK from config.py; existing outputs there are overwritten.
# Build times and search measurements are saved by the individual scripts.
# Fresh timings and HNSW results can differ from the historical report.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"

# Confirm the external evaluator is installed before starting expensive builds.
"$PY" -c 'import config, pathlib; assert pathlib.Path(config.TREC_EVAL).is_file(), "Install trec_eval or set TREC_EVAL; see README.md"'

# 1. BM25: build the index and return the top 100 for every query set.
"$PY" bm25.py build --k1 0.9 --b 0.4
"$PY" bm25.py search --depth 100

# 2-4. Build one HNSW graph, then search at three efSearch settings.
# Search latency is measured one query at a time on one FAISS thread;
# graph construction and the separate batch ranking use the CLI thread default.
"$PY" dense.py build --M 8 --efc 200
"$PY" dense.py search --M 8 --efc 200 --efs 50 100 200 --depth 100 --lat-sample 1000

# 5. Score all BM25 top-1000 candidates by dot product; keep the best 100.
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 "$PY" rerank.py --candidates 1000 --depth 100

# 6-9. Fuse the saved BM25 and HNSW efSearch=200 depth-100 rankings.
# This produces RRF k=10/60/100 AND min-max CombSUM (enabled by default).
# Fusion logs include measured merge time and estimated sequential/parallel totals.
"$PY" fusion.py --sparse bm25 --dense hnsw_M8_efc200_efs200 --k 10 60 100 --depth 100

# Evaluate all nine configurations on DL19, DL20, and dev, then print quality,
# query latency, construction cost, and memory tables from the saved results.
"$PY" evaluate_trec.py
"$PY" experiments.py
