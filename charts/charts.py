import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# ── Paths (repository-relative, resolve via script location) ──────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(REPO_ROOT, "results-run3", "calls.jsonl")
HARD_DATASET = os.path.join(REPO_ROOT, "datasets", "classifier_hard_dataset.json")
GOLD_DATASET = os.path.join(REPO_ROOT, "datasets", "gold_qa_set.json")
OUTDIR = os.path.join(REPO_ROOT, "charts")

with open(LEDGER, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

# ── Basic run-3 checks ────────────────────────────────────────────────────
CLASSIFIERS = [
    "luna",
    "luna_structured",
    "sonnet",
    "sonnet_low",
    "gpt56",
    "jev",
    "jev_fewshot",
    "jev_atomic",
    "jev_atomic_fewshot",
    "jev_parallel",
    "jev_parallel_fewshot",
]

pairs = {(r["question_id"], r["classifier"]) for r in rows}
assert len(pairs) == 1166, f"Expected 1166 unique pairs, got {len(pairs)}"

errors = [r for r in rows if r.get("error")]
assert not errors, f"{len(errors)} error rows found"

for c in CLASSIFIERS:
    arm_rows = [r for r in rows if r["classifier"] == c]
    assert len(arm_rows) == 106, f"{c}: expected 106 rows, got {len(arm_rows)}"

with open(HARD_DATASET, encoding="utf-8") as f:
    hard_data = json.load(f)
hard_ids = {q["id"] for q in hard_data["questions"]}
with open(GOLD_DATASET, encoding="utf-8") as f:
    gold_data = json.load(f)
gold_ids = {q["id"] for q in gold_data["questions"]}

# ── Accuracy derivation ───────────────────────────────────────────────────
hard_questions = {q["id"]: q for q in hard_data["questions"]}


def is_correct(row):
    qid = row["question_id"]
    if qid not in hard_questions:
        return None
    q = hard_questions[qid]
    label = row.get("label")
    return label == q["expected_label"] or (
        q.get("label_certainty") == "contested" and label == q.get("alternative_label")
    )


ACC = {}
for c in CLASSIFIERS:
    arm_hard = [r for r in rows if r["classifier"] == c and r["question_id"] in hard_ids]
    correct = sum(1 for r in arm_hard if is_correct(r) is True)
    ACC[c] = correct

ORDER = list(ACC)

# ── Clear-row set and slice derivation ────────────────────────────────────
# Ambiguous/oos clear rows = hard definitive rows where expected_label is
# ambiguous or out_of_scope, plus gold rows where expected_route is
# ambiguous or out_of_scope respectively.
#
# Gold is derived from expected_route (ground truth), not from LLM predictions.
LLM = {"luna", "luna_structured", "sonnet", "sonnet_low", "gpt56"}

hard_clear_ids = {q["id"] for q in hard_data["questions"] if q.get("label_certainty") == "clear"}
hard_clear_amb = {
    q["id"]
    for q in hard_data["questions"]
    if q.get("label_certainty") == "clear" and q["expected_label"] == "ambiguous"
}
hard_clear_oos = {
    q["id"]
    for q in hard_data["questions"]
    if q.get("label_certainty") == "clear" and q["expected_label"] == "out_of_scope"
}

gold_amb = {q["id"] for q in gold_data["questions"] if q["expected_route"] == "ambiguous"}
gold_oos = {q["id"] for q in gold_data["questions"] if q["expected_route"] == "out_of_scope"}

amb_clear_ids = hard_clear_amb | gold_amb  # 5+2 = 7
oos_clear_ids = hard_clear_oos | gold_oos  # 2+4 = 6

expected_ambiguous = len(amb_clear_ids)
expected_oos = len(oos_clear_ids)

JEV = [c for c in CLASSIFIERS if c.startswith("jev")]
AMB = {}
OOS = {}
for c in JEV:
    amb_rows = [r for r in rows if r["classifier"] == c and r["question_id"] in amb_clear_ids]
    oos_rows = [r for r in rows if r["classifier"] == c and r["question_id"] in oos_clear_ids]
    AMB[c] = sum(1 for r in amb_rows if r.get("label") == "ambiguous")
    OOS[c] = sum(1 for r in oos_rows if r.get("label") == "out_of_scope")


# ── Latency aggregation ───────────────────────────────────────────────────
def pctl(values, p):
    """Return the nearest-rank percentile."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


def agg(arm, field):
    return [r[field] for r in rows if r["classifier"] == arm and r[field] is not None]


cost = {a: sum(agg(a, "list_price_cost_usd")) for a in ORDER}
p50 = {a: pctl(agg(a, "latency_ms"), 0.50) for a in ORDER}
p95 = {a: pctl(agg(a, "latency_ms"), 0.95) for a in ORDER}

# ── Styling ───────────────────────────────────────────────────────────────
INK = "#1a1a1a"
GRID = "#d8d8d8"
C_LLM = "#2f6fb5"
C_JEV = "#c1553b"
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.edgecolor": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

# ── Chart 1 — accuracy vs cost ───────────────────────────────────────────
OFF = {
    "luna": (1.12, 0, "left"),
    "luna_structured": (1.12, 0, "left"),
    "sonnet": (1.12, 0, "left"),
    "sonnet_low": (1.12, 0, "left"),
    "gpt56": (1.12, 0, "left"),
    "jev": (1.15, -0.28, "left"),
    "jev_fewshot": (0.86, 0, "right"),
    "jev_atomic": (0.86, 0.30, "right"),
    "jev_atomic_fewshot": (0.86, 0, "right"),
    "jev_parallel": (1.15, -0.30, "left"),
    "jev_parallel_fewshot": (1.12, 0, "left"),
}
fig, ax = plt.subplots(figsize=(10, 6.2))
for a in ORDER:
    c = C_LLM if a in LLM else C_JEV
    ax.scatter(cost[a], ACC[a], s=95, color=c, zorder=3)
    dx, dy, ha = OFF[a]
    ax.annotate(a, (cost[a] * dx, ACC[a] + dy), va="center", ha=ha, fontsize=10)
ax.set_xscale("log")
ax.set_xlim(0.0016, 3.0)
ax.set_ylim(42.5, 55.5)
ax.set_xlabel("cost per 106 classifications, list price (log scale)")
ax.set_ylabel("correct, hard set (58; accepted alternative counts on contested rows)")
ax.set_title(
    "Hard-set score versus cost of the full 106-question run", loc="left", fontsize=13, pad=14
)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.3f}".rstrip("0").rstrip(".")))
ax.grid(color=GRID, lw=0.6, zorder=0)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.scatter([0.00185], [55.0], s=80, color=C_LLM)
ax.text(0.00205, 55.0, "LLM arms", va="center", fontsize=10)
ax.scatter([0.0043], [55.0], s=80, color=C_JEV)
ax.text(0.00475, 55.0, "Jev arms", va="center", fontsize=10)
fig.text(
    0.125,
    0.02,
    "Across the nine arms repeated in runs 2 and 3, hard-set totals changed by 0–2 of 58. "
    "One repeat is not an uncertainty estimate.",
    fontsize=9,
    color="#555",
)
fig.tight_layout(rect=[0, 0.045, 1, 1])
fig.savefig(os.path.join(OUTDIR, "01-accuracy-vs-cost.png"), dpi=200)

# ── Chart 2 — ambiguous & out_of_scope across jev arms ───────────────────
fig, ax = plt.subplots(figsize=(10, 5.4))
x = range(len(JEV))
w = 0.38
ax.bar(
    [i - w / 2 for i in x],
    [AMB[a] / expected_ambiguous * 100 for a in JEV],
    w,
    color="#c1553b",
    label=f"ambiguous ({expected_ambiguous} rows)",
)
ax.bar(
    [i + w / 2 for i in x],
    [OOS[a] / expected_oos * 100 for a in JEV],
    w,
    color="#e0a37f",
    label=f"out_of_scope ({expected_oos} rows)",
)
ax.axhline(100, color=C_LLM, ls="--", lw=1.2)
ax.text(5.42, 101.5, "every LLM arm: 100% on both", color=C_LLM, fontsize=9.5, ha="right")
ax.set_xticks(list(x))
ax.set_xticklabels([a.replace("jev_", "").replace("jev", "bare") for a in JEV], fontsize=10)
ax.set_ylabel("% correct")
ax.set_ylim(0, 118)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_title(
    "Sequential Noul + Choice-8: ambiguous improved, out_of_scope fell",
    loc="left",
    fontsize=13,
    pad=14,
)
ax.legend(frameon=False, loc="upper left", fontsize=10)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
fig.text(
    0.5,
    0.02,
    f"Clear rows across hard and gold: "
    f"n={expected_ambiguous} ambiguous; n={expected_oos} out_of_scope.",
    fontsize=9,
    color="#555",
    ha="center",
)
fig.tight_layout(rect=[0, 0.045, 1, 1])
fig.savefig(os.path.join(OUTDIR, "02-ambiguous-vs-outofscope.png"), dpi=200)

# ── Chart 3 — latency ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.6))
srt = sorted(ORDER, key=lambda a: p50[a])
ys = range(len(srt))
for i, a in enumerate(srt):
    c = C_LLM if a in LLM else C_JEV
    ax.plot([p50[a], p95[a]], [i, i], color=c, lw=2, alpha=0.35, zorder=2)
    ax.scatter(p50[a], i, color=c, s=70, zorder=3)
    ax.scatter(p95[a], i, color=c, s=28, marker="|", zorder=3)
ax.set_yticks(list(ys))
ax.set_yticklabels(srt, fontsize=10)
ax.set_xlabel("latency, ms — dot = p50, tick = p95")
ax.set_title("Latency per classification", loc="left", fontsize=13, pad=14)
ax.grid(axis="x", color=GRID, lw=0.6)
ax.set_axisbelow(True)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
fig.text(
    0.5,
    0.02,
    "End-to-end latency under run-3 provider conditions, fixed arm-major order, concurrency 4.",
    fontsize=9,
    color="#555",
    ha="center",
)
fig.tight_layout(rect=[0, 0.045, 1, 1])
fig.savefig(os.path.join(OUTDIR, "03-latency.png"), dpi=200)


# ── Chart 4 — confidence histograms ──────────────────────────────────────
def conf(arm):
    return [r["confidence"] for r in rows if r["classifier"] == arm and r["confidence"] is not None]


fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
bins = [i / 10 for i in range(11)]
PANELS = [
    ("luna", "Luna — self-reported"),
    ("sonnet", "Sonnet 5 — self-reported"),
    ("jev", "Bare Jev — Choice-distribution-derived"),
]
for ax, a, t in zip(axes, [p[0] for p in PANELS], [p[1] for p in PANELS], strict=True):
    ax.hist(conf(a), bins=bins, color=C_LLM if a in LLM else C_JEV, edgecolor="white")
    ax.set_title(t, fontsize=11, loc="left")
    ax.set_xlabel("confidence")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
axes[0].set_ylabel("questions")
fig.suptitle(
    "Representative confidence distributions, 106 questions each", fontsize=13, x=0.5, ha="center"
)
fig.text(
    0.5,
    0.02,
    "Confidence semantics differ; these distributions are descriptive and are not "
    "directly comparable as calibrated probabilities.",
    fontsize=9,
    color="#555",
    ha="center",
)
fig.tight_layout(rect=[0, 0.06, 1, 0.93])
fig.savefig(os.path.join(OUTDIR, "04-confidence.png"), dpi=200)

# ── Summary ───────────────────────────────────────────────────────────────
print("ok", {a: (round(cost[a], 4), int(p50[a]), int(p95[a])) for a in ORDER})
print("accuracy:", ACC)
print("ambiguous:", AMB)
print("out_of_scope:", OOS)
print(f"clear denominators: ambiguous={expected_ambiguous}, out_of_scope={expected_oos}")

# Scorer's headline cohort: 88 = 40 non-deterministic hard clear + 48 gold
# hard_clear_ids includes deterministic rows (e.g. GQ-102 caught_by=deterministic)
hard_clear_excl_det = {
    q["id"]
    for q in hard_data["questions"]
    if q.get("label_certainty") == "clear" and q.get("caught_by") != "deterministic"
}
hard_gold_count = len(hard_clear_excl_det)
gold_count = len(gold_data["questions"])
total_clear = hard_gold_count + gold_count  # 40 + 48 = 88
print(f"clear rows: {total_clear} total")
print(f"  ({hard_gold_count} hard clear excl. deterministic + {gold_count} gold)")
print(f"  {len(amb_clear_ids)} amb clear + {len(oos_clear_ids)} oos clear")
