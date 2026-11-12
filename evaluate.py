import argparse
import csv
import glob
import json
import math
import os
import re
import subprocess

import config as C
from common import load_qrels, read_run, write_trec_qrels

GRADED = ["MRR@10", "R@100", "nDCG@10", "nDCG@100"]
BINARY = ["MRR@10", "R@100", "MAP@100"]


# trec_eval
def _parse(out):
    vals = {}
    for line in out.splitlines():
        m, qid, v = line.split()
        if qid == "all":
            vals[m] = float(v)
    return vals


def trec_eval(qrels_path, run_path, level):
    base = [C.TREC_EVAL, "-c", "-l", str(level)]
    a = _parse(subprocess.run(base + ["-M", "10", "-m", "recip_rank", qrels_path, run_path],
                              capture_output=True, text=True, check=True).stdout)
    b = _parse(subprocess.run(base + ["-m", "recall.100", "-m", "ndcg_cut.10,100", "-m", "map",
                                      qrels_path, run_path],
                              capture_output=True, text=True, check=True).stdout)
    return {"MRR@10": a["recip_rank"], "R@100": b["recall_100"], "nDCG@10": b["ndcg_cut_10"],
            "nDCG@100": b["ndcg_cut_100"], "MAP@100": b["map"]}


# main

def _order(path):
    # bm25 first, then hnsw by M, efc, efs (numerically)
    tag = os.path.basename(path).split(".", 1)[1][:-4]
    group = next((i for i, p in enumerate(["bm25", "hnsw"])
                  if tag.startswith(p)), 9)
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", tag)]
    return (group, nums)


def main(check):
    os.makedirs(C.RESULT_DIR, exist_ok=True)
    use_te = os.path.exists(C.TREC_EVAL)
    if not use_te:
        print(f"trec_eval not found at {C.TREC_EVAL}; using the Python implementation")
    rows, maxdiff = [], {}
    for name, qpath in C.QUERY_SETS.items():
        if not os.path.exists(qpath):
            continue
        qrels = load_qrels(qpath)
        qtrec = os.path.join(C.RESULT_DIR, f"qrels.{name}.trec")
        write_trec_qrels(qrels, qtrec)
        level = C.DL_BINARY_REL_LEVEL if name in C.GRADED_SETS else 1
        for rp in sorted(glob.glob(os.path.join(C.RUN_DIR, f"{name}.*.run")), key=_order):
            tag = os.path.basename(rp)[len(name) + 1:-4]
            run = read_run(rp)
            m = trec_eval(qtrec, rp, level) if use_te else py_eval(qrels, run, level)
            row = {"set": name, "system": tag, "n_queries": len(qrels), **m}
            rows.append(row)
    if check:
        print("max |trec_eval - python| per measure:", maxdiff)

    fields = ["set", "system", "n_queries", "MRR@10", "R@100", "nDCG@10", "nDCG@100", "MAP@100"]
    with open(os.path.join(C.RESULT_DIR, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    md = []
    for name in C.QUERY_SETS:
        rs = [r for r in rows if r["set"] == name]
        if not rs:
            continue
        cols = GRADED if name in C.GRADED_SETS else BINARY
        md.append(f"### {name} ({rs[0]['n_queries']} queries)\n")
        md.append("| System | " + " | ".join(cols) + " |")
        md.append("|---" * (len(cols) + 1) + "|")
        for r in rs:
            md.append(f"| {r['system']} | " + " | ".join(
                f"{r[c]:.4f}" if c in r else "–" for c in cols) + " |")
        md.append("")
    md += efficiency_tables()
    with open(os.path.join(C.RESULT_DIR, "tables.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))


def efficiency_tables():
    """Latency (single thread, one query at a time) and index cost, from work/logs."""
    def load(p):
        with open(p) as f:
            return json.load(f)
    md = ["### Query latency (single thread, ms/query)\n",
          "| System | Set | mean | p50 | p99 | QPS |", "|---|---|---|---|---|---|"]

    def add(sysname, setname, st):
        if st:
            md.append(f"| {sysname} | {setname} | {st['mean_ms']:.2f} | {st['p50_ms']:.2f} | "
                      f"{st['p99_ms']:.2f} | {st['qps']:.0f} |")
    for p in sorted(glob.glob(os.path.join(C.LOG_DIR, "search_*.json"))):
        j = load(p)
        tag = os.path.basename(p)[7:-5]
        for s, st in j["sets"].items():
            add(tag, s, st)
    md += ["", "### Index construction and memory\n",
           "| Index | Build time (s) | Index size (MB) | Peak RSS while building (MB) |",
           "|---|---|---|---|"]
    for p in sorted(glob.glob(os.path.join(C.LOG_DIR, "build_*.json"))):
        j = load(p)
        size = j.get("index_disk_mb", 0)
        extra = f" (vectors {j['vectors_mb']:.0f} + graph {j['graph_mb']:.0f})" if "graph_mb" in j else ""
        md.append(f"| {os.path.basename(p)[6:-5]} | {j['build_s']:.1f} | {size:.0f}{extra} | "
                  f"{j['peak_rss_mb']:.0f} |")
    md.append("")
    return md


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    main(ap.parse_args().check)
