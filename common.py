# Shared helpers: file loading, id mapping, run-file writing, timing and memory
import json
import os
import resource
import sys
import time

import h5py
import numpy as np


# loading
def _decode(arr):
    """h5 id arrays may be bytes, str or ints -> list of str."""
    out = []
    for x in arr:
        if isinstance(x, bytes):
            x = x.decode("utf-8")
        elif isinstance(x, np.ndarray):  # e.g. shape (n,1)
            x = x.item()
            x = x.decode("utf-8") if isinstance(x, bytes) else x
        out.append(str(x).strip())
    return out


def load_h5_embeddings(path, id_key=None, emb_key=None):
    # Return (ids:list[str], emb:float32 [n,d]).
    # Key names are auto-detected (an int/bytes 1-D dataset = ids, a 2-D float
    # dataset = embeddings) unless given explicitly.
    
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        if emb_key is None:
            cands = [k for k in keys if f[k].ndim == 2 and f[k].dtype.kind == "f"]
            if len(cands) != 1:
                sys.exit(f"Cannot detect embedding dataset in {path}; keys={keys}. Use --emb-key.")
            emb_key = cands[0]
        if id_key is None:
            cands = [k for k in keys if k != emb_key and f[k].shape[0] == f[emb_key].shape[0]]
            if len(cands) != 1:
                sys.exit(f"Cannot detect id dataset in {path}; keys={keys}. Use --id-key.")
            id_key = cands[0]
        ids = _decode(f[id_key][:])
        emb = np.ascontiguousarray(f[emb_key][:], dtype=np.float32)
    return ids, emb


def load_subset_ids(path):
    # First column of the subset tsv (header lines without a numeric id are skipped)
    ids = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            tok = line.split("\t", 1)[0].strip()
            if tok and tok.isdigit():
                ids.append(tok)
    return ids


def load_queries(*paths):
    # qid -> text from one or more 'qid<TAB>text' files
    q = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) == 2:
                    q[parts[0].strip()] = parts[1]
    return q


def load_qrels(path):
    # Accepts 'qid docid rel' or TREC 'qid 0 docid rel' (tab or space separated).
    # Returns {qid: {docid: rel}}
    qrels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) == 4:
                qid, _, did, rel = p
            elif len(p) == 3:
                qid, did, rel = p
            else:
                continue
            if not rel.lstrip("-").isdigit():  # header
                continue
            qrels.setdefault(qid, {})[did] = int(rel)
    return qrels


def write_trec_qrels(qrels, path):
    # Normalised 4-column qrels so trec_eval always accepts them
    with open(path, "w") as f:
        for qid, docs in qrels.items():
            for did, rel in docs.items():
                f.write(f"{qid} 0 {did} {rel}\n")


# id mapping
class IdMap:
    # passage id (str) -> row index. Uses a dense numpy table when ids are numeric
    def __init__(self, ids):
        self.ids = np.asarray(ids)
        if all(s.isdigit() for s in ids):
            ints = np.asarray(ids, dtype=np.int64)
            self.table = np.full(int(ints.max()) + 1, -1, dtype=np.int32)
            self.table[ints] = np.arange(len(ints), dtype=np.int32)
            self.d = None
        else:
            self.table = None
            self.d = {s: i for i, s in enumerate(ids)}

    def rows(self, ids):
        if self.table is not None:
            a = np.asarray(ids, dtype=np.int64)
            out = np.full(len(a), -1, dtype=np.int64)
            ok = a < len(self.table)
            out[ok] = self.table[a[ok]]
            return out
        return np.array([self.d.get(s, -1) for s in ids], dtype=np.int64)


# run files
def write_run(results, path, tag):
    # results: {qid: [(docid, score), ...]} already sorted by score desc
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for qid, docs in results.items():
            for rank, (did, score) in enumerate(docs, 1):
                f.write(f"{qid} Q0 {did} {rank} {score:.6f} {tag}\n")


def read_run(path):
    run = {}
    with open(path) as f:
        for line in f:
            qid, _, did, _, score, _ = line.split()
            run.setdefault(qid, []).append((did, float(score)))
    return run


# measurement
def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def latency_stats(ms):
    ms = np.asarray(ms, dtype=np.float64)
    if len(ms) == 0:
        return {}
    return {
        "n_queries": int(len(ms)),
        "mean_ms": float(ms.mean()),
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "total_s": float(ms.sum() / 1000),
        "qps": float(1000.0 / ms.mean()) if ms.mean() > 0 else None,
    }


def dir_size_mb(path):
    if os.path.isfile(path):
        return os.path.getsize(path) / 2**20
    total = 0
    for root, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total / 2**20


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t
