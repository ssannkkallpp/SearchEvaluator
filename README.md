# CS 6913 Assignment 3: BM25 vs. HNSW dense retrieval

## Files

| File | What it does |
|---|---|
| `config.py` | All paths and parameters. Edit this first. |
| `bm25.py` | BM25 inverted index over the 1M-passage subset (`build`, `search`) |
| `dense.py` | faiss `IndexHNSWFlat`, inner product (`build`, `search` with several efSearch values) |
| `evaluate.py` | Runs trec_eval on every run file and writes `work/results/tables.md` and `results.csv` |
| `common.py` | Loaders (h5, tsv, queries, qrels), run-file writer, timing and memory helpers |
| `run_all.sh` | The whole pipeline |
| `tests/make_synthetic.py` | Small fake dataset in the course formats, for testing |

## Setup

```bash
pip install numpy scipy scikit-learn h5py faiss-cpu
git clone https://github.com/usnistgov/trec_eval.git && make -C trec_eval
```

Put the Drive files and `collection.tsv` (from assignment 2) in `data/`. Then fix the file names in `config.py`.

## Run

```bash
./run_all.sh
```

or run the steps one at a time:

```bash
python bm25.py build
python bm25.py search
python dense.py build  --M 4 --efc 50
python dense.py search --M 4 --efc 50  --efs 50 100 200
python dense.py build  --M 8 --efc 200
python dense.py search --M 8 --efc 200 --efs 50 100 200
python evaluate.py
```

Outputs:

- `work/runs/{dl19,dl20,dev}.<system>.run`: TREC run files, depth 100
- `work/logs/*.json`: build time, index size, peak RSS, per-query latency
- `work/results/tables.md`: the quality, latency and index-cost tables

## Measures

- **DL 2019 / 2020:** MRR@10, Recall@100, nDCG@10, nDCG@100. For MRR and Recall, a label ≥ 2 counts as relevant (TREC DL convention); change this with `DL_BINARY_REL_LEVEL`.
- **Dev:** MRR@10, Recall@100, MAP@100.
- **trec_eval calls:** `-c -M 10 -m recip_rank`, and `-c -m recall.100 -m ndcg_cut.10,100 -m map`.
- **Latency:** one thread, one query at a time. Run files use batched multi-threaded search.

## Notes

- The vectors take 1M × 384 × 4 B ≈ 1.43 GiB. The HNSW graph adds about 2·M × 4 B per node: about 31 MiB for M=4 and 61 MiB for M=8.
- If efSearch < k, faiss can return fewer than k hits, which lowers Recall@100.

## Test without the real data

```bash
python tests/make_synthetic.py /tmp/synth
A3_DATA=/tmp/synth A3_WORK=/tmp/synth_work ./run_all.sh
```

## Recorded experiment checkpoint

BM25 and HNSW (M=8, efConstruction=200, efSearch=50/100/200) have
been run on the supplied one-million-passage subset. BM25-to-embedding
re-ranking is implemented in `rerank.py`.

Create a virtual environment and install the recorded dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The local `data/` folder requires `collection.tsv`,
`msmarco_passages_subset.tsv`, `msmarco_passages_embeddings_subset.h5`,
`msmarco_queries_dev_eval_embeddings.h5`, `queries.eval.tsv`,
`queries.dev.tsv`, `qrels.eval.one.tsv`, `qrels.eval.two.tsv`, and
`qrels.dev.tsv`. Datasets, indexes, and retrieval runs are excluded from Git.
Paths can be overridden using `A3_DATA` and `A3_WORK`.

### Official evaluation

Install the version used for the recorded results:

```bash
git clone https://github.com/usnistgov/trec_eval.git
git -C trec_eval checkout 35cf096ba6d5ae08ca4803c056b31ed61a56a6d4
make -C trec_eval
make -C trec_eval quicktest
```

This commit reports version `10.0-rc3`. After generating the retrieval runs:

```bash
python evaluate_trec.py
```

Use `evaluate_trec.py` for the official comparison. It calls the external
executable, normalizes qrels into separate generated files, and saves official
scores, raw commands/output, and tables in `work/results/trec_eval/`.
`compare_retrieval.py` contains the earlier standard-library metric calculations;
its scores are retained for comparison, not as the official evaluation.
`evaluate.py` is the older evaluator: its missing-tool fallback and `--check`
option remain incomplete. The original `run_all.sh` still ends with that older
evaluator; use the individual retrieval commands followed by `evaluate_trec.py`
for this checkpoint.

Evaluation averages over all judged queries (`-c`). DL MRR and Recall use
relevance >=2; NDCG uses original grades with linear gains. Dev uses relevance
>=1. Reciprocal rank is restricted to depth 10; Recall and NDCG use their
specified cutoffs. MAP is measured on depth-100 rankings. trec_eval sorts by
saved scores and breaks ties by descending lexicographic passage ID; the earlier
Python comparison follows saved ranks. This explains the small DL20 BM25 NDCG
differences recorded in `differences.csv`.

Tracked `work/logs/` files record build/search performance. Tracked result files
record both the initial Python comparison and official trec_eval evaluation;
normalized copies of the source judgments remain local.

## BM25 candidate re-ranking

`rerank.py` reuses the saved BM25 index, retrieves up to 1,000 candidates per
query, and sorts their supplied embeddings by dot product with the query vector.
It saves the top 100. Equal similarities retain BM25 order in the run file;
`trec_eval` applies its own score-tie ordering during evaluation.

```bash
# Optional small run; outputs stay separate from full evaluation runs.
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python rerank.py --limit 3

# Complete all three query sets, then evaluate all three approaches.
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python rerank.py --candidates 1000
python evaluate_trec.py
```

Use `--candidates 100` to compare a smaller candidate pool. `--depth` defaults
to 100 and must not exceed the candidate count. The script checks query coverage,
embedding dimensions, duplicate embedding IDs, and missing passage embeddings.
Full runs are saved as `work/runs/<set>.rerank_k1000.run`; smoke runs are under
`work/smoke_runs/`. Timing logs include BM25 retrieval, embedding lookup/scoring/
sorting, combined search latency, load time, and peak process RSS. Query latency
excludes loading and run-file serialization. Thread environment settings are
recorded in the log; the commands above restrict the numerical backend to one
thread where those environment variables are supported. No new dependency or
HNSW index is required.
