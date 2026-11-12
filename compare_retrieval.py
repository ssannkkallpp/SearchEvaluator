"""Compare saved BM25/HNSW runs using only the Python standard library.

Run: python compare_retrieval.py
NDCG uses linear gains (the judgment labels), as in trec_eval ndcg_cut.
MAP@100 divides by all known relevant passages, including unretrieved ones.
Missing queries score zero; unjudged passages receive relevance zero.
"""
import csv
import math
from pathlib import Path

import config as C


def load_judgments(path):
    judgments = {}
    with open(path) as handle:
        for line in handle:
            fields = line.split()
            if len(fields) == 4:
                qid, _, did, label = fields
            elif len(fields) == 3:
                qid, did, label = fields
            else:
                continue
            if label.lstrip('-').isdigit():
                judgments.setdefault(qid, {})[did] = int(label)
    if not judgments:
        raise ValueError(f'No judgments in {path}')
    return judgments


def load_ranking(path):
    runs = {}
    with open(path) as handle:
        for line in handle:
            qid, _, did, rank, score, _ = line.split()
            runs.setdefault(qid, []).append((int(rank), did))
    for qid, entries in runs.items():
        entries.sort()
        ids = [did for _, did in entries]
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate passages for {qid} in {path}')
        runs[qid] = ids
    return runs


def query_metrics(judgments, ranking, level):
    relevant = {did for did, label in judgments.items() if label >= level}
    hits = [did in relevant for did in ranking[:100]]
    rr = next((1 / rank for rank, hit in enumerate(hits[:10], 1) if hit), 0.0)
    recall = sum(hits) / len(relevant) if relevant else 0.0
    found = 0
    ap = 0.0
    for rank, hit in enumerate(hits, 1):
        if hit:
            found += 1
            ap += found / rank
    ap = ap / len(relevant) if relevant else 0.0
    ideal = sorted((max(0, label) for label in judgments.values()), reverse=True)
    ndcg = {}
    for cutoff in (10, 100):
        actual = sum(max(0, judgments.get(did, 0)) / math.log2(rank + 1)
                     for rank, did in enumerate(ranking[:cutoff], 1))
        best = sum(label / math.log2(rank + 1)
                   for rank, label in enumerate(ideal[:cutoff], 1))
        ndcg[f'NDCG@{cutoff}'] = actual / best if best else 0.0
    return {'MRR@10': rr, 'Recall@100': recall, **ndcg, 'MAP@100': ap}


def main():
    rows = []
    tables = ['Judgments are unchanged; DL binary relevance threshold = '
              f'{C.DL_BINARY_REL_LEVEL}. NDCG uses original graded labels. '
              'MAP is evaluated at depth 100. Missing queries score zero.\n']
    for name, qpath in C.QUERY_SETS.items():
        judgments = load_judgments(qpath)
        paths = [p for p in Path(C.RUN_DIR).glob(f'{name}.*.run')
                 if p.name.startswith((f'{name}.bm25.', f'{name}.hnsw_'))]
        if not paths:
            raise FileNotFoundError(f'No BM25/HNSW runs for {name}')
        columns = ['MRR@10', 'Recall@100'] + (
            ['NDCG@10', 'NDCG@100'] if name in C.GRADED_SETS else ['MAP@100'])
        tables += [f'\n### {name}: {len(judgments)} judged queries\n',
                   '| System | Queries in run | ' + ' | '.join(columns) + ' |',
                   '|---|---|' + '---|' * len(columns)]
        level = C.DL_BINARY_REL_LEVEL if name in C.GRADED_SETS else 1
        for path in sorted(paths):
            run = load_ranking(path)
            metrics = [query_metrics(docs, run.get(qid, []), level)
                       for qid, docs in judgments.items()]
            means = {key: sum(m[key] for m in metrics) / len(metrics)
                     for key in metrics[0]}
            row = {'set': name, 'system': path.name[len(name) + 1:-4],
                   'judged_queries': len(judgments),
                   'queries_in_run': sum(qid in run for qid in judgments), **means}
            rows.append(row)
            tables.append(f"| {row['system']} | {row['queries_in_run']} | " +
                          ' | '.join(f'{means[key]:.4f}' for key in columns) + ' |')
    output = Path(C.RESULT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'comparison.md').write_text('\n'.join(tables) + '\n')
    with (output / 'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('\n'.join(tables))
    print(f'\nSaved comparison.md and comparison.csv in {output}')


if __name__ == '__main__':
    main()
