# Jev Benchmark

This repository contains the datasets, experiment outputs, and chart generator for a
classification comparison between six Jev configurations and five LLM arms. The evaluation
uses 106 SEC-filing classification questions: 58 hard questions and 48 gold (to another project) questions.

The experiment runner and scorer are not included. This repository preserves the copied
results and provides an offline script for regenerating the charts.

## Included experiment data

- `results-run1`: 742 rows across 7 arms
- `results-run2`: 954 rows across 9 arms
- `results-run3`: 1,166 rows across 11 arms

Each result directory contains:

- `calls.jsonl`: one row per classification call
- `prompts.json`: prompts and structured question definitions
- `runs.jsonl`: run lifecycle and configuration records
- `summary.txt`: per-arm scores, costs, latency, confidence, and disagreements

## Run 3 at a glance

- `jev_parallel_fewshot`: 50 of 58 hard questions
- `luna`: 48 of 58 hard questions
- `sonnet`: 53 of 58 hard questions
- The nine arms repeated in runs 2 and 3 changed by no more than 2 of 58 questions.

The five LLM arms are `luna`, `luna_structured`, `sonnet`, `sonnet_low`, and `gpt56`.
The six Jev arms are `jev`, `jev_fewshot`, `jev_atomic`, `jev_atomic_fewshot`,
`jev_parallel`, and `jev_parallel_fewshot`.

## Repository contents

```text
datasets/
    classifier_hard_dataset.json
    gold_qa_set.json
results-run1/
results-run2/
results-run3/
charts/
    charts.py
    01-accuracy-vs-cost.png
    02-ambiguous-vs-outofscope.png
    03-latency.png
    04-confidence.png
requirements.txt
LICENSE
```

## Generate the charts

Python 3.11 or later is required. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python charts/charts.py
```

The script reads `results-run3/calls.jsonl` and the two datasets, then writes the four PNG
files into `charts/`. Paths are resolved from the script location, so the command also works
when invoked from another directory.

## Interpretation notes

- The atomic and parallel Jev arms were added after earlier results were inspected, so those
  comparisons are exploratory.
- The dataset is a fixed, hand-built set in one domain and is not an uncertainty estimate.
- Costs are reconstructed from recorded usage and list prices; they are not invoice totals.
- Latency reflects the provider and concurrency conditions of each run.
- Confidence values use different semantics across model families and should not be compared
  as if they were the same calibrated probability.

## License

MIT. See `LICENSE`.
