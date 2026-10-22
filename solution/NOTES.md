# Solution Notes

## Approach

This solution implements a deterministic, snapshot-aware knowledge retrieval system over the supplied Dupaco website snapshots.

Key capabilities:
- Historical snapshot preservation for `as_of` queries
- Structured Markdown and table-row extraction
- Intent-aware and qualifier-aware retrieval
- APR/APY, term, balance-tier, jumbo/standard, and personal/business disambiguation
- Authority-based source ranking
- Current vs stale value resolution
- Safe abstention when evidence is insufficient

No external APIs, LLMs, API keys, live website calls, or third-party Python packages are required.

## Evaluation Progress

| Version | Public Evaluation Score | Errors |
|---|---:|---:|
| V1 | 0.4134 | 0 |
| V2 | 0.5841 | 0 |
| V3 | 0.8477 | 0 |
| V4 | **0.8942** | **0** |
| V5 | **0.8942** | **0** |

### Final Evaluation Highlights

- **Best public evaluation score: 0.8942**
- **Accuracy / QA score: 0.8942**
- **Consistency: 1.00**
- **Speed: 1.00**
- **Runtime errors: 0**
- **Current value: 1.00**
- **Entity attribution: 1.00**
- **Should refuse: 1.00**
- **Temporal `as_of`: 1.00**
- **Tier disambiguation: 1.00**
- **Stale trap: 0.96**
- **Branch info: 0.89**
- **Conflict resolution: 0.67**
- **List completeness: 0.50**

V4 and V5 achieved the same best public evaluation score of 0.8942. V4 remains the conservative grading checkpoint because V5 did not provide a measurable improvement.

## Run

`python harness/run_eval.py --submission solution --eval eval/public_eval.jsonl`