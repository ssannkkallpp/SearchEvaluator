"""
Fuse the saved BM25 and HNSW rankings into one ranking.

Methods:
  rrf      Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009):
               score(d) = sum over input rankings of 1 / (k + rank(d))
           Uses ranks only, so BM25 scores and dot products never have to be
           put on a common scale. k = 60 by default, the value from the paper.
  combsum  Score-based baseline for comparison: min-max normalise each ranking's
           scores per query to [0, 1], then add them (a passage missing from a
           ranking contributes 0).

Inputs are the depth-100 run files written by bm25.py and dense.py, so nothing
is re-indexed or re-searched. Each fused ranking keeps the top 100 of the union
of the two lists (at most 200 passages).
"""
import argparse
import json
from pathlib import Path

import numpy as np

import config as C
from common import Timer, latency_stats, load_qrels, read_run, peak_rss_mb, save_json


def ranked(docs):
    # Order exactly as trec_eval does: score descending, ties by passage ID descending.
    # The rank a passage gets here is therefore the rank trec_eval scored it at.
    return [pid for pid, _ in sorted(docs, key=lambda x: (x[1], x[0]), reverse=True)]


def rrf(rankings, k):
    # rankings: list of passage-ID lists, best first. Rank starts at 1, as in the paper.
    fused = {}
    for ranking in rankings:
        for rank, pid in enumerate(ranking, 1):
            fused[pid] = fused.get(pid, 0.0) + 1.0 / (k + rank)
    return fused


def combsum(score_lists):
    # score_lists: list of [(pid, score), ...]. Min-max per list, then sum.
    fused = {}
    for docs in score_lists:
        if not docs:
            continue
        scores = np.array([s for _, s in docs], dtype=np.float64)
        lo, hi = scores.min(), scores.max()
        norm = (scores - lo) / (hi - lo) if hi > lo else np.ones_like(scores)
        for (pid, _), s in zip(docs, norm):
            fused[pid] = fused.get(pid, 0.0) + float(s)
    return fused


def top(fused, depth):
    # Same tie rule as trec_eval, so the rank column agrees with how it is scored.
    return sorted(fused.items(), key=lambda x: (x[1], x[0]), reverse=True)[:depth]


def write_fused(results, path, tag):
    # 10 decimals: RRF sums of nearby ranks differ by ~1e-5, so 6 decimals could
    # create ties that are not real.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for qid, docs in results.items():
            for rank, (pid, score) in enumerate(docs, 1):
                f.write(f'{qid} Q0 {pid} {rank} {score:.10f} {tag}\n')


def input_latency(system, name):
    # Mean single-query latency of an input system, from its search log (if present).
    log = Path(C.LOG_DIR) / f'search_{system}.json'
    if not log.is_file():
        return None
    return json.loads(log.read_text()).get('sets', {}).get(name, {}).get('mean_ms')


def run(sparse, dense, ks, depth, with_combsum):
    if any(k <= 0 for k in ks):
        raise ValueError('RRF k must be positive')
    methods = [(f'rrf_k{k}_{sparse}+{dense}', k) for k in ks]
    if with_combsum:
        methods.append((f'combsum_{sparse}+{dense}', None))
    summaries = {tag: {'system': 'fusion', 'method': 'rrf' if k else 'combsum', 'k': k,
                       'inputs': [sparse, dense], 'depth': depth, 'sets': {}}
                 for tag, k in methods}
    for name, qrels_path in C.QUERY_SETS.items():
        inputs = []
        for system in (sparse, dense):
            path = Path(C.RUN_DIR) / f'{name}.{system}.run'
            if not path.is_file():
                raise FileNotFoundError(f'{path} not found; run bm25.py search and dense.py search first')
            inputs.append(read_run(str(path)))
        qids = list(load_qrels(qrels_path))   # the same queries every other system is scored on
        lists = {qid: [inp.get(qid, []) for inp in inputs] for qid in qids}
        union = [len({pid for docs in lists[q] for pid, _ in docs}) for q in qids]
        bm, hn = input_latency(sparse, name), input_latency(dense, name)
        for tag, k in methods:
            results, ms = {}, []
            for qid in qids:
                with Timer() as t:
                    if k:
                        fused = rrf([ranked(docs) for docs in lists[qid]], k)
                    else:
                        fused = combsum(lists[qid])
                    results[qid] = top(fused, depth)
                ms.append(t.s * 1000)
            write_fused(results, Path(C.RUN_DIR) / f'{name}.{tag}.run', tag)
            stats = latency_stats(ms)            # fusion step only
            fuse = stats.get('mean_ms', 0.0)
            stats.update({
                'avg_union_size': float(np.mean(union)) if union else 0.0,
                'bm25_mean_ms': bm, 'dense_mean_ms': hn,
                # Run one after the other, the total is the sum; run in parallel, the max.
                'sequential_total_mean_ms': bm + hn + fuse if bm and hn else None,
                'parallel_total_mean_ms': max(bm, hn) + fuse if bm and hn else None,
            })
            summaries[tag]['sets'][name] = stats
            print(tag, name, {key: stats.get(key) for key in ('mean_ms', 'avg_union_size')},
                  flush=True)
    for tag, summary in summaries.items():
        summary['peak_rss_mb'] = peak_rss_mb()
        save_json(summary, str(Path(C.LOG_DIR) / f'search_{tag}.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sparse', default='bm25', help='run tag of the lexical input')
    parser.add_argument('--dense', default='hnsw_M8_efc200_efs200', help='run tag of the dense input')
    parser.add_argument('--k', type=int, nargs='+', default=[60], help='RRF constant(s)')
    parser.add_argument('--depth', type=int, default=C.RUN_DEPTH)
    parser.add_argument('--no-combsum', action='store_true', help='skip the CombSUM baseline')
    args = parser.parse_args()
    run(args.sparse, args.dense, args.k, args.depth, not args.no_combsum)
