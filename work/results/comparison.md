Judgments are unchanged; DL binary relevance threshold = 2. NDCG uses original graded labels. MAP is evaluated at depth 100. Missing queries score zero.


### dl19: 43 judged queries

| System | Queries in run | MRR@10 | Recall@100 | NDCG@10 | NDCG@100 |
|---|---|---|---|---|---|
| bm25 | 43 | 0.6159 | 0.5486 | 0.4709 | 0.5340 |
| hnsw_M8_efc200_efs100 | 43 | 0.8283 | 0.6425 | 0.6987 | 0.6558 |
| hnsw_M8_efc200_efs200 | 43 | 0.8283 | 0.6366 | 0.6987 | 0.6544 |
| hnsw_M8_efc200_efs50 | 43 | 0.8283 | 0.6519 | 0.6994 | 0.6549 |

### dl20: 54 judged queries

| System | Queries in run | MRR@10 | Recall@100 | NDCG@10 | NDCG@100 |
|---|---|---|---|---|---|
| bm25 | 54 | 0.6858 | 0.6117 | 0.5028 | 0.5231 |
| hnsw_M8_efc200_efs100 | 54 | 0.7797 | 0.7339 | 0.6627 | 0.6377 |
| hnsw_M8_efc200_efs200 | 54 | 0.7797 | 0.7289 | 0.6627 | 0.6344 |
| hnsw_M8_efc200_efs50 | 54 | 0.7612 | 0.7126 | 0.6500 | 0.6259 |

### dev: 1000 judged queries

| System | Queries in run | MRR@10 | Recall@100 | MAP@100 |
|---|---|---|---|---|
| bm25 | 1000 | 0.4040 | 0.7945 | 0.4044 |
| hnsw_M8_efc200_efs100 | 1000 | 0.5427 | 0.8599 | 0.5414 |
| hnsw_M8_efc200_efs200 | 1000 | 0.5603 | 0.8909 | 0.5592 |
| hnsw_M8_efc200_efs50 | 1000 | 0.5072 | 0.8111 | 0.5058 |
