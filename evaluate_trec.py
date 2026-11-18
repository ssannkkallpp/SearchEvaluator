"""Evaluate saved BM25, HNSW, and re-ranking runs with official trec_eval."""
import csv
import subprocess
from pathlib import Path

import config as C
from common import load_qrels


def main():
    if not Path(C.TREC_EVAL).is_file():
        raise FileNotFoundError(f'trec_eval executable not found: {C.TREC_EVAL}. Set TREC_EVAL to its path.')
    output = Path(C.RESULT_DIR) / 'trec_eval'
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, source in C.QUERY_SETS.items():
        qrels = output / f'{name}.qrels'
        judgments = load_qrels(source)
        if not judgments:
            raise ValueError(f"No judgments found in {source}")
        with qrels.open('w') as f:
            for qid, docs in judgments.items():
                for did, rel in docs.items():
                    f.write(f'{qid} 0 {did} {rel}\n')
        for run in sorted(Path(C.RUN_DIR).glob(f'{name}.*.run')):
            system = run.name[len(name) + 1:-4]
            if not system.startswith(('bm25', 'hnsw', 'rerank', 'rrf', 'combsum')):
                continue
            level = C.DL_BINARY_REL_LEVEL if name in C.GRADED_SETS else 1
            calls = [(['-l', str(level), '-M', '10', '-m', 'recip_rank'],
                      {'recip_rank': 'MRR@10'}),
                     (['-l', str(level), '-m', 'recall.100'],
                      {'recall_100': 'Recall@100'})]
            if name in C.GRADED_SETS:
                calls.append((['-m', 'ndcg_cut.10,100'],
                              {'ndcg_cut_10': 'NDCG@10', 'ndcg_cut_100': 'NDCG@100'}))
            else:
                calls.append((['-M', '100', '-m', 'map'], {'map': 'MAP@100'}))
            row = {'set': name, 'system': system}
            raw = []
            for flags, mapping in calls:
                command = [C.TREC_EVAL, '-c', *flags, str(qrels), str(run)]
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                raw.extend(['# ' + ' '.join(command), result.stdout.rstrip()])
                for line in result.stdout.splitlines():
                    metric, qid, value = line.split()
                    if qid == 'all' and metric in mapping:
                        row[mapping[metric]] = float(value)
            (output / f'{name}.{system}.txt').write_text('\n'.join(raw) + '\n')
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f'No retrieval runs found in {C.RUN_DIR}')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (output / 'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    md = []
    for name in C.QUERY_SETS:
        cols = ['MRR@10', 'Recall@100'] + (['NDCG@10', 'NDCG@100']
                if name in C.GRADED_SETS else ['MAP@100'])
        md += [f'\n### {name}\n', '| System | ' + ' | '.join(cols) + ' |',
               '|---|' + '---|' * len(cols)]
        for row in rows:
            if row['set'] == name:
                md.append('| ' + row['system'] + ' | ' +
                          ' | '.join(f'{row[c]:.4f}' for c in cols) + ' |')
    (output / 'tables.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))
    print(f'\nSaved official results to {output}')


if __name__ == '__main__':
    main()
