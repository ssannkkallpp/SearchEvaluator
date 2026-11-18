"""Print all results: quality (trec_eval), query latency, and index build/memory.

Reads what the pipeline already wrote; runs nothing and writes nothing.

    python experiments.py
"""
import csv
import json
from pathlib import Path

import config as C

DENSE = 'hnsw_M8_efc200_efs200'
SYSTEMS = [  # (label, run tag) in report order
    ('BM25', 'bm25'),
    ('HNSW (8, 200, 50)', 'hnsw_M8_efc200_efs50'),
    ('HNSW (8, 200, 100)', 'hnsw_M8_efc200_efs100'),
    ('HNSW (8, 200, 200)', 'hnsw_M8_efc200_efs200'),
    ('Re-rank BM25 top-1000', 'rerank_k1000'),
    ('RRF, k = 10', f'rrf_k10_bm25+{DENSE}'),
    ('RRF, k = 60', f'rrf_k60_bm25+{DENSE}'),
    ('RRF, k = 100', f'rrf_k100_bm25+{DENSE}'),
    ('CombSUM (min-max)', f'combsum_bm25+{DENSE}'),
]


def log(name):
    path = Path(C.LOG_DIR) / name
    return json.loads(path.read_text()) if path.is_file() else None


def show(title, headers, rows):
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    line = lambda r: '  '.join(str(x).ljust(w) if i == 0 else str(x).rjust(w)
                               for i, (x, w) in enumerate(zip(r, widths)))
    print(f'\n{title}\n' + line(headers) + '\n' + '  '.join('-' * w for w in widths))
    for r in rows:
        print(line(r))


def ms(st):
    if not st or st.get('mean_ms') is None:
        return ['-'] * 4
    f = lambda k: f'{st[k]:.2f}' if st.get(k) is not None else '-'
    return [f('mean_ms'), f('p50_ms'), f('p99_ms'), f"{1000 / st['mean_ms']:,.0f}"]


def main():
    # Quality
    with (Path(C.RESULT_DIR) / 'trec_eval' / 'results.csv').open() as f:
        quality = {(r['set'], r['system']): r for r in csv.DictReader(f)}
    for name in C.QUERY_SETS:
        cols = ['MRR@10', 'Recall@100'] + (['NDCG@10', 'NDCG@100'] if name in C.GRADED_SETS else ['MAP@100'])
        q = lambda tag, c: quality.get((name, tag), {}).get(c)
        rows = [[label] + [f'{float(q(tag, c)):.4f}' if q(tag, c) else '-' for c in cols] for label, tag in SYSTEMS]
        show(f'=== Quality: {name} ===', ['System'] + cols, rows)

    # Latency (single thread, ms per query)
    rr = log('search_rerank_k1000.json')
    rf = log(f'search_rrf_k60_bm25+{DENSE}.json')
    for name in C.QUERY_SETS:
        rs = rr and rr['sets'].get(name)
        fs = (rf and rf['sets'].get(name)) or {}
        rows = [['BM25'] + ms((log('search_bm25.json') or {}).get('sets', {}).get(name))]
        rows += [[f'HNSW (8, 200, {e})'] + ms((log(f'search_hnsw_M8_efc200_efs{e}.json') or {}).get('sets', {}).get(name))
                 for e in (50, 100, 200)]
        rows += [['Re-rank: BM25 stage'] + ms(rs and rs.get('bm25')),
                 ['Re-rank: dense stage'] + ms(rs and rs.get('rerank')),
                 ['Re-rank: total'] + ms(rs),
                 ['RRF k=60: fusion only'] + ms(fs),
                 ['RRF k=60: total, sequential'] + ms({'mean_ms': fs.get('sequential_total_mean_ms')}),
                 ['RRF k=60: total, parallel'] + ms({'mean_ms': fs.get('parallel_total_mean_ms')})]
        show(f'=== Latency: {name} (ms/query, 1 thread) ===', ['System', 'Mean', 'p50', 'p99', 'QPS'], rows)

    # Index build and memory
    b, h = log('build_bm25.json'), log('build_hnsw_M8_efc200.json')
    rows = []
    if b:
        rows.append(['BM25', f"{b['build_s']:.1f}", f"{b.get('index_disk_mb') or b['index_mem_mb']:,.0f}",
                     f"{b['peak_rss_mb']:,.0f}"])
    if h:
        rows.append(['HNSW (8, 200)', f"{h['build_s']:.1f}",
                     f"{h['index_disk_mb']:,.0f} ({h['vectors_mb']:,.0f} vectors + {h['graph_mb']:,.0f} graph)",
                     f"{h['peak_rss_mb']:,.0f}"])
    if rr:
        rows.append(['Re-rank (search process)', '-', '-', f"{rr['peak_rss_mb']:,.0f}"])
    show('=== Index build and memory ===', ['Index', 'Build (s)', 'Size (MB)', 'Peak RSS (MB)'], rows)


if __name__ == '__main__':
    main()
