"""Retrieve BM25 candidates, then rank them by embedding dot product.

Examples:
    python rerank.py
    python rerank.py --candidates 100
    python rerank.py --limit 3  # smoke test; writes separately named outputs
"""
import argparse
import os
from pathlib import Path

import numpy as np

import config as C
from bm25 import BM25
from common import (Timer, latency_stats, load_h5_embeddings, load_qrels,
                    load_queries, peak_rss_mb, save_json, write_run)


def embedding_lookup(ids, vectors, label):
    if vectors.shape != (len(ids), 384):
        raise ValueError(f'{label}: expected {len(ids)} x 384 embeddings, got {vectors.shape}')
    lookup = {pid: row for row, pid in enumerate(ids)}
    if len(lookup) != len(ids):
        raise ValueError(f'{label}: duplicate embedding IDs')
    if not np.isfinite(vectors).all():
        raise ValueError(f'{label}: embeddings contain non-finite values')
    return lookup


def rerank_candidates(ids, query_vector, passage_vectors, passage_rows, depth):
    # Gather vectors in BM25 order so every score stays attached to its passage ID.
    missing = [pid for pid in ids if pid not in passage_rows]
    if missing:
        raise ValueError(f'Missing passage embeddings: {missing[:5]}')
    rows = [passage_rows[pid] for pid in ids]
    scores = passage_vectors[rows] @ query_vector
    if not np.isfinite(scores).all():
        raise ValueError('Dot products contain non-finite values')
    order = np.argsort(-scores, kind='stable')[:depth]
    return ids[order], scores[order]


def run(candidates, depth, limit):
    tag = f'rerank_k{candidates}' + (f'_smoke{limit}' if limit else '')
    # Load once; no model training or HNSW index is needed.
    with Timer() as loading:
        bm = BM25()
        pids, passages = load_h5_embeddings(C.PASSAGE_EMB_H5, C.H5_ID_KEY, C.H5_EMB_KEY)
        qids, queries = load_h5_embeddings(C.QUERY_EMB_H5, C.H5_ID_KEY, C.H5_EMB_KEY)
        passage_rows = embedding_lookup(pids, passages, 'Passages')
        query_rows = embedding_lookup(qids, queries, 'Queries')
        texts = load_queries(*C.QUERIES_TSV)
        sets = {name: list(load_qrels(path)) for name, path in C.QUERY_SETS.items()}
        for name, selected in sets.items():
            missing = [qid for qid in selected if qid not in texts or qid not in query_rows]
            if missing:
                raise ValueError(f'{name}: missing query text or embedding for {missing[:5]}')
        missing = [pid for pid in bm.doc_ids if pid not in passage_rows]
        if missing:
            raise ValueError(f'BM25 corpus has missing passage embeddings: {missing[:5]}')
    summary = {'system': 'rerank', 'candidates': candidates, 'depth': depth,
               'load_s': loading.s, 'limit': limit,
               'thread_environment': {key: os.environ.get(key) for key in
                                      ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
               'sets': {}}
    for name, selected in sets.items():
        selected = selected[:limit] if limit else selected
        results, bm_ms, rerank_ms, total_ms, counts = {}, [], [], [], []
        for qid in selected:
            with Timer() as total:
                with Timer() as retrieval:
                    ids, _ = bm.search(texts[qid], candidates)
                with Timer() as ranking:
                    ranked_ids, scores = rerank_candidates(
                        ids, queries[query_rows[qid]], passages, passage_rows, depth)
            results[qid] = list(zip(ranked_ids.tolist(), scores.tolist()))
            bm_ms.append(retrieval.s * 1000)
            rerank_ms.append(ranking.s * 1000)
            total_ms.append(total.s * 1000)
            counts.append(len(ids))
        # Keep smoke-test results separate from complete runs and official evaluation.
        folder = Path(C.WORK_DIR) / 'smoke_runs' if limit else Path(C.RUN_DIR)
        write_run(results, str(folder / f'{name}.{tag}.run'), tag)
        stats = latency_stats(total_ms)
        stats.update({'bm25': latency_stats(bm_ms), 'rerank': latency_stats(rerank_ms),
                      'avg_candidates': float(np.mean(counts)) if counts else 0})
        summary['sets'][name] = stats
        print(name, stats, flush=True)
    summary['peak_rss_mb'] = peak_rss_mb()
    save_json(summary, str(Path(C.LOG_DIR) / f'search_{tag}.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=int, default=1000)
    parser.add_argument('--depth', type=int, default=100)
    parser.add_argument('--limit', type=int, default=0, help='queries per set for a smoke test')
    args = parser.parse_args()
    if not 0 < args.depth <= args.candidates or args.limit < 0:
        parser.error('require 0 < depth <= candidates and limit >= 0')
    run(args.candidates, args.depth, args.limit)
