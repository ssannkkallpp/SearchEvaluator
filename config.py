import os

DATA_DIR = os.environ.get("A3_DATA", "data")      # folder with the Google Drive files
WORK_DIR = os.environ.get("A3_WORK", "work")      # indexes, runs, logs and results go here

COLLECTION_TSV = os.path.join(DATA_DIR, "collection.tsv")

SUBSET_IDS_TSV = os.path.join(DATA_DIR, "msmarco_passages_subset.tsv")
PASSAGE_EMB_H5 = os.path.join(DATA_DIR, "msmarco_passages_embeddings_subset.h5")
QUERY_EMB_H5 = os.path.join(DATA_DIR, "msmarco_queries_dev_eval_embeddings.h5")
QUERIES_TSV = [os.path.join(DATA_DIR, "queries.eval.tsv"),
               os.path.join(DATA_DIR, "queries.dev.tsv")]

# Dataset names inside the h5 files. None = auto-detect (1-D ids + 2-D float matrix).
H5_ID_KEY = None
H5_EMB_KEY = None

# Evaluation sets: name -> qrels file
QUERY_SETS = {
    "dl19": os.path.join(DATA_DIR, "qrels.eval.one.tsv"),  # TREC DL 2019, graded 0-3
    "dl20": os.path.join(DATA_DIR, "qrels.eval.two.tsv"),  # TREC DL 2020, graded 0-3
    "dev":  os.path.join(DATA_DIR, "qrels.dev.tsv"),       # MS MARCO dev, binary
}
GRADED_SETS = {"dl19", "dl20"}   # NDCG for these, MAP for the rest

DL_BINARY_REL_LEVEL = 2

# Retrieval depth written to every run file (all metrics are computed at <= 100).
RUN_DEPTH = 100

# BM25 parameters: MS MARCO defaults
BM25_K1 = 0.9
BM25_B = 0.4

# Keep the external tool outside the project; override this path on other machines.
TREC_EVAL = os.environ.get(
    "TREC_EVAL",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "search_tools", "trec_eval", "trec_eval"),
)

BM25_DIR = os.path.join(WORK_DIR, "bm25_index")
HNSW_DIR = os.path.join(WORK_DIR, "hnsw")
RUN_DIR = os.path.join(WORK_DIR, "runs")
LOG_DIR = os.path.join(WORK_DIR, "logs")
RESULT_DIR = os.path.join(WORK_DIR, "results")
