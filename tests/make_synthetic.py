"""Generate a small synthetic dataset in the exact formats the assignment uses,
so the whole pipeline can be tested before the real data arrives.

    python tests/make_synthetic.py OUT_DIR
"""
import os
import sys

import h5py
import numpy as np

rng = np.random.default_rng(0)
out = sys.argv[1]
os.makedirs(out, exist_ok=True)

D, T, S = 384, 40, 4                         # dim, topics, subtopics per topic
V = 4000
core = rng.choice(V, size=(T, S, 15), replace=False if T * S * 15 <= V else True)
syn = rng.permutation(V)                      # word w has synonym syn[w]
bg_p = 1.0 / np.arange(1, V + 1) ** 1.1
bg_p /= bg_p.sum()
c_topic = rng.normal(size=(T, D)).astype(np.float32)
c_sub = rng.normal(size=(T, S, D)).astype(np.float32) * 0.7


def words(t, s, n, syn_rate=0.0):
    k = n // 2
    w = list(rng.choice(core[t, s], size=k)) + list(rng.choice(V, size=n - k, p=bg_p))
    w = [syn[x] if rng.random() < syn_rate else x for x in w]
    return " ".join(f"w{x}" for x in rng.permutation(w))


# collection: 12k passages, ids not contiguous; subset = 8k of them
N = 12000
pids = np.sort(rng.choice(10**6, size=N, replace=False))
pt, ps = rng.integers(T, size=N), rng.integers(S, size=N)
with open(f"{out}/collection.tsv", "w") as f:
    for i in range(N):
        f.write(f"{pids[i]}\t{words(pt[i], ps[i], rng.integers(20, 60))}\n")
sub = np.sort(rng.choice(N, size=8000, replace=False))
with open(f"{out}/msmarco_passages_subset.tsv", "w") as f:
    for i in sub:
        f.write(f"{pids[i]}\n")
pe = c_topic[pt[sub]] + c_sub[pt[sub], ps[sub]] + rng.normal(size=(len(sub), D)).astype(np.float32)
with h5py.File(f"{out}/msmarco_passages_embeddings_subset.h5", "w") as f:
    f.create_dataset("id", data=np.array([str(p).encode() for p in pids[sub]]))
    f.create_dataset("embedding", data=pe.astype(np.float32))

# queries: eval (ids 1000..) for DL19/DL20, dev (ids 50000..)
subset_pos = {int(pids[i]): j for j, i in enumerate(sub)}


def make_queries(start, n, syn_rate):
    qs = []
    for j in range(n):
        t, s = rng.integers(T), rng.integers(S)
        qs.append((start + j, t, s, words(t, s, rng.integers(3, 8), syn_rate)))
    return qs


evalq = make_queries(1000, 60, 0.5)
devq = make_queries(50000, 300, 0.5)
with open(f"{out}/queries.eval.tsv", "w") as f:
    f.writelines(f"{q}\t{txt}\n" for q, _, _, txt in evalq)
with open(f"{out}/queries.dev.tsv", "w") as f:
    f.writelines(f"{q}\t{txt}\n" for q, _, _, txt in devq)
allq = evalq + devq
qe = np.stack([c_topic[t] + c_sub[t, s] + rng.normal(size=D) * 0.8 for _, t, s, _ in allq])
with h5py.File(f"{out}/msmarco_queries_dev_eval_embeddings.h5", "w") as f:
    f.create_dataset("id", data=np.array([q for q, *_ in allq], dtype=np.int64))
    f.create_dataset("embedding", data=qe.astype(np.float32))

sub_t, sub_s, sub_p = pt[sub], ps[sub], pids[sub]


def graded(q, t, s):
    same = np.where((sub_t == t) & (sub_s == s))[0]
    near = np.where((sub_t == t) & (sub_s != s))[0]
    far = rng.choice(len(sub), 10)
    lab = {}
    for i in rng.choice(same, min(8, len(same)), replace=False):
        lab[sub_p[i]] = int(rng.choice([2, 3]))
    for i in rng.choice(near, min(6, len(near)), replace=False):
        lab[sub_p[i]] = 1
    for i in far:
        lab.setdefault(sub_p[i], 0)
    return lab


with open(f"{out}/qrel.eval.one.tsv", "w") as f:            # TREC 4-column
    for q, t, s, _ in evalq[:30]:
        for p, r in graded(q, t, s).items():
            f.write(f"{q} 0 {p} {r}\n")
    f.write(f"{evalq[30][0]} 0 {sub_p[0]} 1\n")                # topic with no label >= 2
with open(f"{out}/qrel.eval.two.tsv", "w") as f:            # 3-column, tab separated
    for q, t, s, _ in evalq[31:]:
        for p, r in graded(q, t, s).items():
            f.write(f"{q}\t{p}\t{r}\n")
with open(f"{out}/qrels.dev.tsv", "w") as f:                # MS MARCO style "qid 0 pid 1"
    for q, t, s, _ in devq:
        same = np.where((sub_t == t) & (sub_s == s))[0]
        for i in rng.choice(same, min(2, len(same)), replace=False):
            f.write(f"{q}\t0\t{sub_p[i]}\t1\n")
print("synthetic data written to", out)
