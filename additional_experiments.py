"""Run partial reranking and query-length comparisons using existing indexes.

    OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python additional_experiments.py

Every invocation creates a new directory under WORK_DIR/additional_experiments.
No existing runs are read and no index is rebuilt or written.
"""
import argparse
from collections import Counter
from datetime import datetime
import csv
import json
import os
from pathlib import Path
import pickle
import subprocess
import tempfile
import uuid

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'hw3-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import faiss
import numpy as np

import config as C
from bm25 import BM25, tokenize
from common import load_h5_embeddings, load_queries, load_qrels, load_subset_ids
from rerank import embedding_lookup, rerank_candidates

KS = list(range(10, 101, 10))
LABELS = {'bm25': 'BM25', 'hnsw': 'HNSW (efSearch=200)',
          'rerank1000': 'Dense rerank (1,000 candidates)'}


def partial_order(ids, scores, k):
    """Reorder the first k only; preserve both membership and the entire tail."""
    n = min(k, len(ids))
    order = np.concatenate((np.argsort(-scores[:n], kind='stable'),
                            np.arange(n, len(ids))))
    return ids[order]


def write_ranking(path, rankings):
    # trec_eval ignores the rank column. Strictly descending synthetic scores
    # preserve the intended order, including the untouched BM25 tail.
    with path.open('w') as f:
        for qid, ids in rankings.items():
            for rank, pid in enumerate(ids, 1):
                f.write(f'{qid} Q0 {pid} {rank} {len(ids)-rank+1} experiment\n')


def evaluate(name, tag, rankings, judgments, output, commands):
    path = output / 'runs' / f'{name}.{tag}.run'
    write_ranking(path, rankings)
    qrels = output / f'{name}.qrels'
    level = C.DL_BINARY_REL_LEVEL if name in C.GRADED_SETS else 1
    calls = [(['-l', str(level), '-M', '10', '-m', 'recip_rank'],
              {'recip_rank': 'MRR@10'}),
             (['-l', str(level), '-m', 'recall.100'], {'recall_100': 'Recall@100'})]
    calls += [(['-m', 'ndcg_cut.10,100'], {'ndcg_cut_10': 'NDCG@10', 'ndcg_cut_100': 'NDCG@100'})] if name in C.GRADED_SETS else [(['-M', '100', '-m', 'map'], {'map': 'MAP@100'})]
    per_query = {qid: {} for qid in judgments}
    raw = []
    for flags, mapping in calls:
        command = [C.TREC_EVAL, '-c', '-q', *flags, str(qrels), str(path)]
        commands.append(command)
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        raw.append(result.stdout)
        for line in result.stdout.splitlines():
            metric, qid, value = line.split()
            if metric in mapping and qid in per_query:
                per_query[qid][mapping[metric]] = float(value)
        # Explicit zeros retain judged queries with no hits.
        for values in per_query.values():
            for metric in mapping.values():
                values.setdefault(metric, 0.0)
    (output / 'evaluation' / f'{name}.{tag}.txt').write_text('\n'.join(raw))
    return per_query


def save_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(name, metrics, summary, groups, output):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, metric in zip(axes.flat, metrics):
        values = [r for r in summary if r['metric'] == metric]
        ax.plot([r['k'] for r in values], [r['mean'] for r in values],
                'o-', label='Rerank first k of fixed BM25 top 100', color='#176b87')
        for key, color in [('bm25', '#a65d24'), ('hnsw', '#65489b')]:
            ax.axhline(values[0][key], linestyle='--', color=color, label=LABELS[key])
        ax.set(title=metric, xlabel='Number of leading passages reranked (k)', ylabel='Mean score', xticks=KS)
        ax.grid(alpha=.2)
    if len(metrics) < 4:
        axes.flat[-1].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=1)
    fig.suptitle(f'{name.upper()}: partial reranking — fixed top-100 membership', fontsize=14)
    for ext in ('png', 'pdf'):
        fig.savefig(output / 'plots' / f'{name}_partial_reranking.{ext}', dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    lengths = sorted({r['length'] for r in groups})
    x = np.arange(len(lengths))
    for ax, metric in zip(axes.flat, metrics):
        for i, (system, label) in enumerate(LABELS.items()):
            rows = [next(r for r in groups if r['length'] == length and r['system'] == system and r['metric'] == metric) for length in lengths]
            ax.bar(x + (i - 1) * .25, [r['mean'] for r in rows], width=.25, label=label)
        counts = [next(r['n_queries'] for r in groups if r['length'] == length) for length in lengths]
        ax.set(title=metric, ylabel='Mean score', xlabel='Query length (tokens)',
               xticks=x, xticklabels=[f'{length} tokens\n(n={count})' for length, count in zip(lengths, counts)], ylim=(0, 1))
        ax.grid(axis='y', alpha=.2)
    if len(metrics) < 4:
        axes.flat[-1].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3)
    fig.suptitle(f'{name.upper()}: two most common query lengths', fontsize=14)
    for ext in ('png', 'pdf'):
        fig.savefig(output / 'plots' / f'{name}_query_length.{ext}', dpi=220)
    plt.close(fig)


def run(threads):
    output = Path(C.WORK_DIR) / 'additional_experiments' / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    for folder in ('runs', 'evaluation', 'plots'):
        (output / folder).mkdir()
    print(f'Output: {output.resolve()}', flush=True)
    manifest = {'status': 'running', 'threads': threads, 'ks': KS, 'depth': 100,
                'index_root': str(Path(C.WORK_DIR).resolve()), 'commands': [], 'selected_lengths': {}}
    def save_manifest():
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    save_manifest()
    try:
        if not Path(C.TREC_EVAL).is_file():
            raise FileNotFoundError(C.TREC_EVAL)
        bm = BM25()
        with open(Path(C.BM25_DIR) / 'meta.pkl', 'rb') as f:
            meta = pickle.load(f)
        assert (meta['k1'], meta['b']) == (0.9, 0.4), 'Unexpected BM25 settings'
        pids, passages = load_h5_embeddings(C.PASSAGE_EMB_H5, C.H5_ID_KEY, C.H5_EMB_KEY)
        qids, queries = load_h5_embeddings(C.QUERY_EMB_H5, C.H5_ID_KEY, C.H5_EMB_KEY)
        passage_rows = embedding_lookup(pids, passages, 'Passages')
        query_rows = embedding_lookup(qids, queries, 'Queries')
        index = faiss.read_index(str(Path(C.HNSW_DIR) / 'hnsw_M8_efc200.faiss'))
        row_ids = np.load(Path(C.HNSW_DIR) / 'row_ids.npy')
        subset = set(load_subset_ids(C.SUBSET_IDS_TSV))
        assert len(set(bm.doc_ids)) == bm.N == len(subset)
        assert set(bm.doc_ids) == subset == set(pids)
        assert index.ntotal == len(row_ids) == len(pids) and index.d == 384
        assert np.array_equal(row_ids, np.asarray(pids))
        assert index.metric_type == faiss.METRIC_INNER_PRODUCT
        assert index.hnsw.nb_neighbors(1) == 8 and index.hnsw.efConstruction == 200
        for row in np.linspace(0, index.ntotal - 1, 100, dtype=int):
            assert np.array_equal(index.reconstruct(int(row)), passages[row])
        faiss.omp_set_num_threads(threads)
        index.hnsw.efSearch = 200
        texts = load_queries(*C.QUERIES_TSV)
        all_summary, all_groups, all_per_query, length_counts = [], [], [], []
        for name, source in C.QUERY_SETS.items():
            judgments = load_qrels(source)
            assert judgments and all(q in texts and q in query_rows for q in judgments)
            with (output / f'{name}.qrels').open('w') as f:
                for qid, docs in judgments.items():
                    for pid, rel in docs.items():
                        f.write(f'{qid} 0 {pid} {rel}\n')
            counts = Counter(len(tokenize(texts[q])) for q in judgments)
            selected = sorted(sorted(counts, key=lambda length: (-counts[length], length))[:2])
            manifest['selected_lengths'][name] = {str(length): counts[length] for length in selected}
            length_counts.extend({'set': name, 'length': length, 'n_queries': count, 'selected': length in selected} for length, count in sorted(counts.items()))
            rankings = {tag: {} for tag in [*LABELS, *[f'partial{k}' for k in KS]]}
            for qid in judgments:
                ids, _ = bm.search(texts[qid], 1000)
                base = ids[:100]
                query = queries[query_rows[qid]]
                dense_scores = passages[[passage_rows[pid] for pid in base]] @ query
                rankings['bm25'][qid] = base
                rankings['rerank1000'][qid] = rerank_candidates(ids, query, passages, passage_rows, 100)[0]
                _, positions = index.search(query[None, :], 100)
                rankings['hnsw'][qid] = row_ids[positions[0][positions[0] >= 0]]
                for k in KS:
                    ordered = partial_order(base, dense_scores, k)
                    assert set(ordered) == set(base) and np.array_equal(ordered[k:], base[k:])
                    rankings[f'partial{k}'][qid] = ordered
            print(f'{name}: generated fresh rankings for {len(judgments)} queries; lengths {selected}', flush=True)
            evaluations = {tag: evaluate(name, tag, ranks, judgments, output, manifest['commands']) for tag, ranks in rankings.items()}
            metrics = list(next(iter(evaluations['bm25'].values())))
            means = {tag: {metric: float(np.mean([values[metric] for values in per.values()])) for metric in metrics} for tag, per in evaluations.items()}
            summary, groups = [], []
            for k in KS:
                for metric in metrics:
                    assert metric != 'Recall@100' or all(evaluations[f'partial{k}'][q][metric] == evaluations['bm25'][q][metric] for q in judgments)
                    summary.append({'set': name, 'k': k, 'metric': metric, 'mean': means[f'partial{k}'][metric],
                                    'bm25': means['bm25'][metric], 'hnsw': means['hnsw'][metric],
                                    'delta_vs_bm25': means[f'partial{k}'][metric] - means['bm25'][metric]})
            for length in selected:
                members = [q for q in judgments if len(tokenize(texts[q])) == length]
                for tag in LABELS:
                    for metric in metrics:
                        groups.append({'set': name, 'length': length, 'n_queries': len(members), 'system': tag,
                                       'metric': metric, 'mean': float(np.mean([evaluations[tag][q][metric] for q in members]))})
            for tag, per in evaluations.items():
                for qid, values in per.items():
                    for metric, value in values.items():
                        all_per_query.append({'set': name, 'system': tag, 'qid': qid, 'length': len(tokenize(texts[qid])), 'metric': metric, 'value': value})
            all_summary.extend(summary)
            all_groups.extend(groups)
            plot_results(name, metrics, summary, groups, output)
            print(f'{name}: evaluation and plots complete; Recall@100 invariance verified.', flush=True)
            save_manifest()
        for filename, rows in [('partial_reranking.csv', all_summary), ('query_length.csv', all_groups),
                               ('per_query.csv', all_per_query), ('length_counts.csv', length_counts)]:
            save_csv(output / filename, rows)
        manifest['status'] = 'complete'
    except Exception as error:
        manifest.update(status='failed', error=str(error))
        raise
    finally:
        save_manifest()
    print(f'Finished: {output.resolve()}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    run(args.threads)
