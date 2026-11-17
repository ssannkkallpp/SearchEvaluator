# BM25, dense retrieval, and re-ranking

Compare three retrieval approaches on the supplied one-million-passage subset:
BM25 word matching, HNSW embedding search, and BM25 candidates re-ranked by
embedding dot product. All quality evaluation uses official `trec_eval`.

## Core files

| File | Responsibility |
|---|---|
| `config.py` | Data/output paths and retrieval parameters |
| `common.py` | Data loaders, TREC output, timing and memory helpers |
| `bm25.py` | Build and search the subset BM25 index |
| `dense.py` | Build and search a FAISS HNSW index using dot product |
| `rerank.py` | Re-rank BM25 candidates using supplied embeddings |
| `fusion.py` | Fuse saved BM25 and HNSW runs (RRF + CombSUM baseline) |
| `evaluate_trec.py` | Evaluate saved rankings and generate quality tables |
| `run_all.sh` | Rebuild and run all three systems, then evaluate |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Keep these nine supplied files under `data/`:

- `collection.tsv`
- `msmarco_passages_subset.tsv`
- `msmarco_passages_embeddings_subset.h5`
- `msmarco_queries_dev_eval_embeddings.h5`
- `queries.eval.tsv`, `queries.dev.tsv`
- `qrels.eval.one.tsv`, `qrels.eval.two.tsv`, `qrels.dev.tsv`

`A3_DATA` and `A3_WORK` override the default `data/` and `work/` paths.
The assignment specification remains local at `docs/assignment_spec.md`.

The external evaluator checkout lives outside this project, at
`../search_tools/trec_eval/`. To install it on another machine:

```bash
mkdir -p ../search_tools
git clone https://github.com/usnistgov/trec_eval.git ../search_tools/trec_eval
git -C ../search_tools/trec_eval checkout 35cf096ba6d5ae08ca4803c056b31ed61a56a6d4
make -C ../search_tools/trec_eval
make -C ../search_tools/trec_eval quicktest
```

This commit reports version `10.0-rc3`. Set `TREC_EVAL` to use another executable
location; `config.py` otherwise resolves the sibling tools directory automatically.

## Run individual steps

```bash
python bm25.py build
python bm25.py search
python dense.py build --M 8 --efc 200
python dense.py search --M 8 --efc 200 --efs 50 100 200
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python rerank.py --candidates 1000
python fusion.py --k 10 60 100
python evaluate_trec.py
```

Existing indexes can be reused: omit the build commands. To rebuild and run the
whole pipeline, activate the environment and run `./run_all.sh`.

Re-ranking returns the top 100 of up to 1,000 BM25 candidates by default.
`--candidates` and `--depth` are configurable, with depth <= candidates.
Equal dot products retain BM25 order in the saved ranking. Missing embeddings
produce an error. Optional `--limit 3` runs a smoke test with separate outputs.

## Outputs and evaluation

| Folder | Contents |
|---|---|
| `work/bm25_index/` | Saved BM25 index |
| `work/hnsw/` | Saved HNSW index and passage-ID mapping |
| `work/runs/` | Full-system TREC rankings |
| `work/logs/` | Build/search timing and memory logs |
| `work/results/trec_eval/` | Official CSV scores, Markdown tables, raw commands/output, normalized qrels |

DL19/DL20 use MRR@10, Recall@100, NDCG@10 and NDCG@100. Dev uses MRR@10,
Recall@100 and MAP on the available depth-100 rankings. DL binary relevance
uses labels >=2; dev uses >=1. NDCG retains original graded labels and linear
gains. All judged queries are included in averages (`-c`). trec_eval sorts by
saved score and breaks ties using descending lexicographic passage IDs.

Timing excludes some loading/serialization work: BM25 build timing excludes
index saving; HNSW build timing measures vector insertion. Re-ranking logs
separate BM25 time, embedding lookup/scoring/sorting, and combined latency.
Peak RSS is process memory including temporary structures, not just index size.

Datasets, indexes, environments and retrieval runs remain local and are excluded
from Git. All generated work/ artifacts, including logs and evaluation results, remain local and are excluded from Git.
Optional experiment history and superseded code were moved outside the project
to `../search_engines_hw3_archive/2026-11-12/`; its inventory records original paths
and checksums. Those snapshots are not dependencies of the core code.
