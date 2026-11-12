"""Evaluate saved BM25/HNSW rankings with official trec_eval; compare Python scores."""
import csv
import subprocess
from pathlib import Path

import config as C
from compare_retrieval import load_judgments


def main():
    output = Path(C.RESULT_DIR) / 'trec_eval'
    output.mkdir(parents=True, exist_ok=True)
    previous_path = Path(C.RESULT_DIR) / 'comparison.csv'
    previous = {}
    if previous_path.exists():
        with previous_path.open() as f:
            previous = {(r['set'], r['system']): r for r in csv.DictReader(f)}
    rows, deltas = [], []
    for name, source in C.QUERY_SETS.items():
        qrels = output / f'{name}.qrels'
        judgments = load_judgments(source)
        with qrels.open('w') as f:
            for qid, docs in judgments.items():
                for did, rel in docs.items():
                    f.write(f'{qid} 0 {did} {rel}\n')
        for run in sorted(Path(C.RUN_DIR).glob(f'{name}.*.run')):
            system = run.name[len(name) + 1:-4]
            if not system.startswith(('bm25', 'hnsw')):
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
            old = previous.get((name, system))
            if old:
                for metric, value in row.items():
                    if metric not in ('set', 'system'):
                        deltas.append({'set': name, 'system': system, 'metric': metric,
                                       'python': float(old[metric]), 'trec_eval': value,
                                       'difference': value - float(old[metric])})
    for filename, records in [('results.csv', rows), ('differences.csv', deltas)]:
        if records:
            fields = list(dict.fromkeys(k for row in records for k in row))
            with (output / filename).open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(records)
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
    if deltas:
        print('\nMaximum absolute difference:', max(abs(d['difference']) for d in deltas))
        print('Differences > 0.0001:')
        for d in deltas:
            if abs(d['difference']) > 0.0001:
                print(d)
    print(f'\nSaved official results and comparisons to {output}')


if __name__ == '__main__':
    main()
