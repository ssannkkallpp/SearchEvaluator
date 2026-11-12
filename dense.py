"""Dense retrieval with a faiss HNSW index (inner product).

    python dense.py build --M 8 --efc 200          # build + save one HNSW graph
    python dense.py search --M 8 --efc 200 --efs 50 100 200

The embeddings were trained for dot-product similarity, so every index uses
METRIC_INNER_PRODUCT and vectors are NOT normalised.
"""
import argparse
import os

import faiss
import numpy as np

import config as C
from common import (Timer, latency_stats, load_h5_embeddings, load_qrels,
                    peak_rss_mb, save_json, write_run)


def index_path(M, efc):
    return os.path.join(C.HNSW_DIR, f"hnsw_M{M}_efc{efc}.faiss")


def load_passages():
    ids, emb = load_h5_embeddings(C.PASSAGE_EMB_H5, C.H5_ID_KEY, C.H5_EMB_KEY)
    print(f"passage embeddings: {emb.shape}")
    return np.asarray(ids), emb


def load_query_embeddings():
    ids, emb = load_h5_embeddings(C.QUERY_EMB_H5, C.H5_ID_KEY, C.H5_EMB_KEY)
    return {q: i for i, q in enumerate(ids)}, emb

# builds HNSW index. it takes raw continious embeddings, constructs an interconnected
# multi layer graph, save the index and mapping variables on disk and tracks performance
# benchmarks like generation time and memory usage
def build(M, efc, threads):
    # configuration
    os.makedirs(C.HNSW_DIR, exist_ok=True)
    faiss.omp_set_num_threads(threads)
    ids, emb = load_passages()
    # the graph plus the full, uncompressed ("flat") vectors, so every distance computed is exact.
    # since the embeddings were trained for dot product similarit, vectors are not normalized
    index = faiss.IndexHNSWFlat(emb.shape[1], M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = efc
    with Timer() as t:
        index.add(emb)
    path = index_path(M, efc)
    faiss.write_index(index, path)
    # serializes the graph structure and vector weights into a specialized compact binary file
    np.save(os.path.join(C.HNSW_DIR, "row_ids.npy"), ids)
    vec_mb = emb.nbytes / 2**20
    disk_mb = os.path.getsize(path) / 2**20
    stats = {"system": "hnsw", "M": M, "efConstruction": efc, "build_s": t.s,
             "build_threads": threads, "n_vectors": int(emb.shape[0]), "dim": int(emb.shape[1]),
             "index_disk_mb": disk_mb, "vectors_mb": vec_mb, "graph_mb": disk_mb - vec_mb,
             "peak_rss_mb": peak_rss_mb()}
    save_json(stats, os.path.join(C.LOG_DIR, f"build_hnsw_M{M}_efc{efc}.json"))
    print(stats)


def search_sets(index, row_ids, tag, depth, threads, lat_sample, set_params=None):
    # Batch-search every query set (for the run files), then time single queries
    q2row, qemb = load_query_embeddings()
    out = {}
    for name, qpath in C.QUERY_SETS.items():
        if not os.path.exists(qpath):
            continue
        qids = [q for q in load_qrels(qpath) if q in q2row]
        Q = qemb[[q2row[q] for q in qids]]
        if set_params:
            set_params()
        faiss.omp_set_num_threads(threads)
        with Timer() as tb:
            D, I = index.search(Q, depth)
        res = {}
        for qi, qid in enumerate(qids):
            ok = I[qi] >= 0              # HNSW may return < depth hits when efSearch < depth
            res[qid] = list(zip(row_ids[I[qi][ok]].tolist(), D[qi][ok].tolist()))
        write_run(res, os.path.join(C.RUN_DIR, f"{name}.{tag}.run"), tag)
        # per-query latency: one query at a time, single thread
        faiss.omp_set_num_threads(1)
        lat = []
        for qi in range(min(lat_sample, len(qids))):
            with Timer() as t:
                index.search(Q[qi:qi + 1], depth)
            lat.append(t.s * 1000)
        st = latency_stats(lat)
        st["batch_total_s"] = tb.s
        st["batch_threads"] = threads
        st["avg_hits_returned"] = float((I >= 0).sum(1).mean())
        out[name] = st
        print(tag, name, st)
    return out


def run_search(M, efc, efs_list, depth, threads, lat_sample):
    with Timer() as tl:
        index = faiss.read_index(index_path(M, efc))
    row_ids = np.load(os.path.join(C.HNSW_DIR, "row_ids.npy"))
    for efs in efs_list:
        tag = f"hnsw_M{M}_efc{efc}_efs{efs}"

        def setp(efs=efs):
            index.hnsw.efSearch = efs
        sets = search_sets(index, row_ids, tag, depth, threads, lat_sample, setp)
        save_json({"system": "hnsw", "M": M, "efConstruction": efc, "efSearch": efs,
                   "load_s": tl.s, "sets": sets, "peak_rss_mb": peak_rss_mb()},
                  os.path.join(C.LOG_DIR, f"search_{tag}.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "search"])
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--efs", type=int, nargs="+", default=[50, 100, 200])
    ap.add_argument("--depth", type=int, default=C.RUN_DEPTH)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--lat-sample", type=int, default=1000,
                    help="queries per set timed one-at-a-time on 1 thread")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.M, a.efc, a.threads)
    else:
        run_search(a.M, a.efc, a.efs, a.depth, a.threads, a.lat_sample)
