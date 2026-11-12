"""BM25 over the 1M-passage subset.

The index is a term-major (CSC) sparse matrix of precomputed BM25 impact scores
w(t,d) = idf(t) * tf*(k1+1) / (tf + k1*(1-b+b*|d|/avgdl)), i.e. an inverted index
whose postings already hold the per-document BM25 contribution. Query scoring is
term-at-a-time (disjunctive / OR semantics) with a dense accumulator.

    python bm25.py build                # filter collection.tsv to the subset and index it
    python bm25.py search               # run every query set, write runs + latency logs
"""
import argparse
import os
import pickle
import re

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

import config as C
from common import (Timer, dir_size_mb, latency_stats, load_qrels, load_queries,
                    load_subset_ids, peak_rss_mb, save_json, write_run)

TOKEN_RE = re.compile(r"(?u)\b\w+\b")

def tokenize(text):
    return TOKEN_RE.findall(text.lower())


# build
def iter_subset_passages(collection, subset, doc_ids):
    with open(collection, encoding="utf-8") as f:
        for line in f:
            pid, _, text = line.rstrip("\n").partition("\t")
            if pid in subset:
                doc_ids.append(pid)
                yield text


def build(k1, b):
    os.makedirs(C.BM25_DIR, exist_ok=True)
    subset = set(load_subset_ids(C.SUBSET_IDS_TSV))
    print(f"subset ids: {len(subset):,}")
    doc_ids = []
    with Timer() as t:
        # configure CountVectorizer 
        cv = CountVectorizer(lowercase=True, token_pattern=TOKEN_RE.pattern, dtype=np.int32)
        # process collection.tsv from COO matrix format to compressed CSR matrix
        X = cv.fit_transform(iter_subset_passages(C.COLLECTION_TSV, subset, doc_ids)).tocsr()
        X.sort_indices()
        N = X.shape[0]
        missing = len(subset) - N
        if missing:
            print(f"WARNING: {missing:,} subset ids not found in collection.tsv")
        dl = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
        avgdl = float(dl.mean())
        # vectorized collection statistics
        df = np.bincount(X.indices, minlength=X.shape[1]).astype(np.float32)
        idf = np.log(1.0 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
        rows = np.repeat(np.arange(N), np.diff(X.indptr))
        tf = X.data.astype(np.float32)
        # one BM25 weight per nonzero entry
        w = idf[X.indices] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[rows] / avgdl))
        # flip to term major: all the entries of one column, meaning one term are stored together
        # in a compressed sparse matrix
        W = sp.csr_matrix((w.astype(np.float32), X.indices, X.indptr), shape=X.shape).tocsc()
        del X, rows, tf, w
    # save it to the disk
    sp.save_npz(os.path.join(C.BM25_DIR, "impacts.npz"), W, compressed=False)
    with open(os.path.join(C.BM25_DIR, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab": cv.vocabulary_, "doc_ids": doc_ids, "k1": k1, "b": b,
                     "avgdl": avgdl}, f, protocol=4)
    stats = {
        "system": "bm25", "n_docs": N, "n_terms": W.shape[1], "n_postings": int(W.nnz),
        "avgdl": avgdl, "k1": k1, "b": b, "build_s": t.s,
        "index_mem_mb": (W.data.nbytes + W.indices.nbytes + W.indptr.nbytes) / 2**20,
        "index_disk_mb": dir_size_mb(C.BM25_DIR), "peak_rss_mb": peak_rss_mb(),
    }
    save_json(stats, os.path.join(C.LOG_DIR, "build_bm25.json"))
    print(stats)

# search
class BM25:
    def __init__(self, index_dir=None):
        index_dir = index_dir or C.BM25_DIR
        self.W = sp.load_npz(os.path.join(index_dir, "impacts.npz")).tocsc()
        with open(os.path.join(index_dir, "meta.pkl"), "rb") as f:
            m = pickle.load(f)
        self.vocab = m["vocab"]
        self.doc_ids = np.asarray(m["doc_ids"])
        self.N = self.W.shape[0]

    def search(self, text, k):
        # Returns (doc_ids, scores) sorted by score desc; only docs matching >=1 term
        counts = {}
        for tok in tokenize(text):
            tid = self.vocab.get(tok)
            if tid is not None:
                counts[tid] = counts.get(tid, 0) + 1
        if not counts:
            return self.doc_ids[:0], np.zeros(0, np.float32)
        W, ip = self.W, self.W.indptr
        docs, wts = [], []
        for tid, qtf in counts.items():
            s, e = ip[tid], ip[tid + 1]
            docs.append(W.indices[s:e])
            wts.append(W.data[s:e] * qtf if qtf > 1 else W.data[s:e])
        docs = np.concatenate(docs)
        # np.bincount(docids, weights) adds up the weights that land on same docid.
        # # score per passage
        acc = np.bincount(docs, weights=np.concatenate(wts), minlength=self.N)
        kk = min(k, self.N)
        # unordered top k, O(N)
        top = np.argpartition(-acc, kk - 1)[:kk]
        top = top[acc[top] > 0]                 # idf > 0, so 0 means "no term matched"
        top = top[np.lexsort((top, -acc[top]))]  # score desc, ties by doc order
        return self.doc_ids[top], acc[top].astype(np.float32)


# for each query set: it loads the qrels, keeps the qids that have query text 
# and times search() on each one
def run_search(depth):
    queries = load_queries(*C.QUERIES_TSV)
    with Timer() as t_load:
        bm = BM25()
    summary = {"system": "bm25", "load_s": t_load.s, "sets": {}}
    for name, qpath in C.QUERY_SETS.items():
        if not os.path.exists(qpath):
            print(f"skip {name}: {qpath} not found")
            continue
        qids = [q for q in load_qrels(qpath) if q in queries]
        res, lat = {}, []
        for qid in qids:
            with Timer() as t:
                ids, sc = bm.search(queries[qid], depth)
            lat.append(t.s * 1000)
            res[qid] = list(zip(ids.tolist(), sc.tolist()))
        write_run(res, os.path.join(C.RUN_DIR, f"{name}.bm25.run"), "bm25")
        summary["sets"][name] = latency_stats(lat)
        print(name, summary["sets"][name])
    summary["peak_rss_mb"] = peak_rss_mb()
    save_json(summary, os.path.join(C.LOG_DIR, "search_bm25.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "search"])
    ap.add_argument("--k1", type=float, default=C.BM25_K1)
    ap.add_argument("--b", type=float, default=C.BM25_B)
    ap.add_argument("--depth", type=int, default=C.RUN_DEPTH)
    a = ap.parse_args()
    build(a.k1, a.b) if a.cmd == "build" else run_search(a.depth)
