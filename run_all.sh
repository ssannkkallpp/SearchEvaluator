#!/usr/bin/env bash
# Rebuild and run all three systems, then evaluate their saved rankings.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"

# Confirm the external evaluator is installed before starting expensive builds.
"$PY" -c 'import config, pathlib; assert pathlib.Path(config.TREC_EVAL).is_file(), "Install trec_eval or set TREC_EVAL; see README.md"'

"$PY" bm25.py build
"$PY" bm25.py search
"$PY" dense.py build --M 8 --efc 200
"$PY" dense.py search --M 8 --efc 200 --efs 50 100 200
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 "$PY" rerank.py --candidates 1000
"$PY" fusion.py --k 10 60 100
"$PY" evaluate_trec.py
