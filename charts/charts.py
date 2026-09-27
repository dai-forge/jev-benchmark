import json
import math
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
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

# ── Run-2 ledger checks & article output dir ──────────────────────────────
RUN2_LEDGER_PATH = os.path.join(REPO_ROOT, "results-run2", "calls.jsonl")
with open(RUN2_LEDGER_PATH, encoding="utf-8") as f:
    rows2 = [json.loads(line) for line in f]
ARTICLE_OUTDIR = os.path.join(REPO_ROOT, "charts", "article")
os.makedirs(ARTICLE_OUTDIR, exist_ok=True)
for a in ["jev", "jev_fewshot", "jev_atomic", "jev_atomic_fewshot"]:
    arm_rows = [r for r in rows2 if r["classifier"] == a]
    assert len(arm_rows) == 106, f"run2 {a}: expected 106 rows, got {len(arm_rows)}"
r2_errors = [r for r in rows2 if r.get("error")]
assert not r2_errors, f"run2: {len(r2_errors)} error rows found"

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


def plot_article_accuracy_cost(outfile):
    article_arms = [a for a in ORDER if a not in {"jev_atomic", "jev_parallel"}]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    for a in article_arms:
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
        0.5,
        0.02,
        "Shows 9 of 11 configurations. All 11 are in charts/01-accuracy-vs-cost.png in the "
        "jev-benchmark repo.",
        fontsize=9,
        color="#555",
        ha="center",
    )
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig(outfile, dpi=200)


plot_article_accuracy_cost(os.path.join(ARTICLE_OUTDIR, "01-accuracy-vs-cost.png"))

# ── Chart 2 — ambiguous & out_of_scope across jev arms ───────────────────


def plot_amb_vs_oos(
    ledger_path,
    arms,
    outfile,
    title,
    footer,
    bar_labels,
    xtick_labels=None,
):
    """Plot ambiguous vs out_of_scope bar chart for any ledger."""
    with open(ledger_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    amb_clear_ids = hard_clear_amb | gold_amb
    oos_clear_ids = hard_clear_oos | gold_oos
    n_amb = len(amb_clear_ids)
    n_oos = len(oos_clear_ids)

    amb_counts = {}
    oos_counts = {}
    for a in arms:
        amb_counts[a] = sum(
            1
            for r in rows
            if r["classifier"] == a and r["question_id"] in amb_clear_ids and r.get("label") == "ambiguous"
        )
        oos_counts[a] = sum(
            1
            for r in rows
            if r["classifier"] == a and r["question_id"] in oos_clear_ids and r.get("label") == "out_of_scope"
        )

    if xtick_labels is None:
        xtick_labels = [a.replace("jev_", "").replace("jev", "bare") for a in arms]

    llm_in_arms = [a for a in arms if a in LLM]
    llm_all_ok = all(
        (
            sum(
                1
                for r in rows
                if r["classifier"] == a and r["question_id"] in amb_clear_ids and r.get("label") == "ambiguous"
            )
            == n_amb
            and sum(
                1
                for r in rows
                if r["classifier"] == a and r["question_id"] in oos_clear_ids and r.get("label") == "out_of_scope"
            )
            == n_oos
        )
        for a in llm_in_arms
    )

    fig, ax = plt.subplots(figsize=(10, 5.4))
    x = range(len(arms))
    w = 0.38
    ax.bar(
        [i - w / 2 for i in x],
        [amb_counts[a] / n_amb * 100 for a in arms],
        w,
        color="#c1553b",
        label=f"ambiguous ({n_amb} rows)",
    )
    ax.bar(
        [i + w / 2 for i in x],
        [oos_counts[a] / n_oos * 100 for a in arms],
        w,
        color="#e0a37f",
        label=f"out_of_scope ({n_oos} rows)",
    )
    ax.axhline(100, color=C_LLM, ls="--", lw=1.2)
    ax.text(5.42, 101.5, "every LLM arm: 100% on both", color=C_LLM, fontsize=9.5, ha="right")
    ax.set_xticks(list(x))
    ax.set_xticklabels(xtick_labels, fontsize=10)
    ax.set_ylabel("% correct")
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title(title, loc="left", fontsize=13, pad=14)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    if bar_labels:
        for i, a in enumerate(arms):
            ax.text(i, amb_counts[a] / n_amb * 100 + 1.2, f"{amb_counts[a]}/{n_amb}", ha="center", va="bottom", fontsize=8)
            ax.text(i, oos_counts[a] / n_oos * 100 + 1.2, f"{oos_counts[a]}/{n_oos}", ha="center", va="bottom", fontsize=8)

    fig.text(
        0.5,
        0.02,
        footer,
        fontsize=9,
        color="#555",
        ha="center",
    )
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig(outfile, dpi=200)


# --- repo chart (all 6 jev arms, run 3, bar_labels off) ---
plot_amb_vs_oos(
    LEDGER,
    JEV,
    os.path.join(OUTDIR, "02-ambiguous-vs-outofscope.png"),
    "Sequential Noul + Choice-8: ambiguous improved, out_of_scope fell",
    f"Clear rows across hard and gold: n={expected_ambiguous} ambiguous; n={expected_oos} out_of_scope.",
    bar_labels=False,
)

# --- article chart (4 jev arms, run 2, bar_labels on) ---
RUN2_LEDGER = os.path.join(REPO_ROOT, "results-run2", "calls.jsonl")
article_arms_2 = ["jev", "jev_fewshot", "jev_atomic", "jev_atomic_fewshot"]
plot_amb_vs_oos(
    RUN2_LEDGER,
    article_arms_2,
    os.path.join(ARTICLE_OUTDIR, "02-ambiguous-vs-outofscope.png"),
    "Run 2: asking 'is it ambiguous?' first fixed one label and broke another",
    "Run 2 ledger. Clear rows across hard and gold: n=7 ambiguous, n=6 out_of_scope. All six Jev arms, run 3: charts/02-ambiguous-vs-outofscope.png in the jev-benchmark repo.",
    bar_labels=True,
    xtick_labels=article_arms_2,
)

# Verify article chart plotted counts
EXPECTED_ARTICLE = {
    "jev": (1, 6),
    "jev_fewshot": (3, 6),
    "jev_atomic": (5, 2),
    "jev_atomic_fewshot": (5, 6),
}
for a, (ea, eo) in EXPECTED_ARTICLE.items():
    amb_count = sum(
        1 for r in rows2 if r["classifier"] == a and r["question_id"] in amb_clear_ids and r.get("label") == "ambiguous"
    )
    oos_count = sum(
        1 for r in rows2 if r["classifier"] == a and r["question_id"] in oos_clear_ids and r.get("label") == "out_of_scope"
    )
    assert amb_count == ea and oos_count == eo, (
        f"run2 article {a}: expected ({ea}, {eo}), got ({amb_count}, {oos_count})"
    )

# ── Chart 3 — latency ────────────────────────────────────────────────────


def plot_latency(arms, outfile, footer, p50_labels):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    srt = sorted(arms, key=lambda a: p50[a])
    ys = range(len(srt))
    for i, a in enumerate(srt):
        c = C_LLM if a in LLM else C_JEV
        ax.plot([p50[a], p95[a]], [i, i], color=c, lw=2, alpha=0.35, zorder=2)
        ax.scatter(p50[a], i, color=c, s=70, zorder=3)
        ax.scatter(
            p95[a],
            i,
            color=c,
            s=70 if p50_labels else 28,
            marker="*" if p50_labels else "|",
            zorder=3,
        )
        if p50_labels:
            ax.text(
                p50[a],
                i + 0.35,
                f"{p50[a]:,.0f} ms",
                va="bottom",
                ha="center",
                fontsize=7.5,
                color=INK,
            )
    ax.set_yticks(list(ys))
    ax.set_yticklabels(srt, fontsize=10)
    p95_symbol = "star" if p50_labels else "tick"
    ax.set_xlabel(f"latency, ms — dot = p50, {p95_symbol} = p95")
    ax.set_title("Latency per classification", loc="left", fontsize=13, pad=14)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    wrapped_footer = textwrap.fill(footer, width=115)
    fig.text(
        0.5,
        0.02,
        wrapped_footer,
        fontsize=9,
        color="#555",
        ha="center",
    )
    bottom = 0.085 if "\n" in wrapped_footer else 0.045
    fig.tight_layout(rect=[0, bottom, 1, 1])
    fig.savefig(outfile, dpi=200)


# --- repo chart (all 11 arms, p50_labels off) ---
plot_latency(
    CLASSIFIERS,
    os.path.join(OUTDIR, "03-latency.png"),
    "End-to-end latency under run-3 provider conditions, fixed arm-major order, concurrency 4.",
    p50_labels=False,
)

# --- article chart (9 arms, p50_labels on) ---
JEX = {"jev_atomic", "jev_parallel"}
article_arms_3 = [a for a in CLASSIFIERS if a not in JEX]
plot_latency(
    article_arms_3,
    os.path.join(ARTICLE_OUTDIR, "03-latency.png"),
    "End-to-end latency under run-3 provider conditions, fixed arm-major order, concurrency 4. Shows 9 of 11 configurations; all 11 in charts/03-latency.png in the jev-benchmark repo.",
    p50_labels=True,
)


# ── Chart 4 — confidence histograms ──────────────────────────────────────


def is_correct_all(row):
    """Correctness for all 106 rows (hard + gold)."""
    qid = row["question_id"]
    label = row.get("label")
    if qid in hard_questions:
        q = hard_questions[qid]
        if label == q["expected_label"]:
            return True
        if q.get("label_certainty") == "contested" and label == q.get("alternative_label"):
            return True
        return False
    if qid in gold_ids:
        q = gold_questions[qid]
        return label == q["expected_route"]
    return None


gold_questions = {q["id"]: q for q in gold_data["questions"]}


def correct_any(row):
    """Score a run-3 row against its hard- or gold-set answer."""
    qid = row["question_id"]
    label = row.get("label")
    if qid in hard_questions:
        q = hard_questions[qid]
        return label == q["expected_label"] or (
            q.get("label_certainty") == "contested" and label == q.get("alternative_label")
        )
    if qid in gold_questions:
        return label == gold_questions[qid]["expected_route"]
    raise KeyError(f"Unknown question_id: {qid}")


# Assert the article claims before any article rendering begins.
article_confidence_rows = {}
for arm, expected_wrong in [("luna", 11), ("sonnet", 6), ("jev_parallel_fewshot", 9)]:
    arm_rows = [r for r in rows if r["classifier"] == arm]
    assert len(arm_rows) == 106, f"{arm}: expected 106 rows, got {len(arm_rows)}"
    assert all(is_correct_all(r) is not None for r in arm_rows), f"{arm}: row outside hard and gold sets"
    assert all(r.get("confidence") is not None for r in arm_rows), f"{arm}: missing confidence"
    wrong = sum(1 for r in arm_rows if is_correct_all(r) is False)
    assert wrong == expected_wrong, f"{arm}: expected {expected_wrong} wrong, got {wrong}"
    article_confidence_rows[arm] = arm_rows

luna_misses_high = sum(
    1
    for r in article_confidence_rows["luna"]
    if is_correct_all(r) is False and r["confidence"] >= 0.86
)
assert luna_misses_high == 8, f"luna misses >= 0.86: expected 8, got {luna_misses_high}"

sonnet_misses = [r for r in article_confidence_rows["sonnet"] if is_correct_all(r) is False]
assert all(r["confidence"] <= 0.85 for r in sonnet_misses), "sonnet: not all misses <= 0.85"

jpf_rows = article_confidence_rows["jev_parallel_fewshot"]
jpf_expected_bins = [
    (1, 0, 0),
    (1, 1, 1),
    (2, 2, 0),
    (2, 1, 2),
    (5, 1, 0),
    (3, 2, 0),
    (80, 2, 0),
]
jpf_actual_bins = []
for bin_index in range(3, 10):
    low = bin_index / 10
    high = (bin_index + 1) / 10
    bin_rows = [
        r
        for r in jpf_rows
        if low <= r["confidence"] < high or (bin_index == 9 and r["confidence"] == 1.0)
    ]
    jpf_actual_bins.append(
        (
            sum(r["decided_by"] == "choice" and is_correct_all(r) is True for r in bin_rows),
            sum(r["decided_by"] == "choice" and is_correct_all(r) is False for r in bin_rows),
            sum(r["decided_by"] == "noul" for r in bin_rows),
        )
    )
assert jpf_actual_bins == jpf_expected_bins, (
    f"jev_parallel_fewshot bins: expected {jpf_expected_bins}, got {jpf_actual_bins}"
)
assert sum(sum(counts) for counts in jpf_actual_bins) == 106, (
    f"jev_parallel_fewshot bins: expected total 106, got "
    f"{sum(sum(counts) for counts in jpf_actual_bins)}"
)
jpf_noul_rows = [r for r in jpf_rows if r["decided_by"] == "noul"]
assert {r["question_id"] for r in jpf_noul_rows} == {"GQ-083", "GQ-084", "GQ-085"}, (
    "jev_parallel_fewshot Noul rows: expected GQ-083, GQ-084, GQ-085, got "
    f"{sorted(r['question_id'] for r in jpf_noul_rows)}"
)
assert all(is_correct_all(r) is True for r in jpf_noul_rows), (
    "jev_parallel_fewshot: not all Noul-decided rows are correct"
)
assert all(r["confidence"] == r["noul_probability"] for r in jpf_noul_rows), (
    "jev_parallel_fewshot: Noul-decided confidence differs from Noul probability"
)


def plot_confidence(panels, outfile, split_correct):
    bins = [i / 10 for i in range(11)]

    if split_correct:
        fig, axes = plt.subplots(
            2,
            3,
            figsize=(11, 4.8),
            sharex="col",
            sharey="row",
            gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.08},
        )
        upper_axes, lower_axes = axes
        for upper, lower, (a, title) in zip(upper_axes, lower_axes, panels, strict=True):
            arm_rows = article_confidence_rows[a]
            if a == "jev_parallel_fewshot":
                correct_confs = [
                    r["confidence"]
                    for r in arm_rows
                    if r["decided_by"] == "choice" and is_correct_all(r) is True
                ]
                wrong_confs = [
                    r["confidence"]
                    for r in arm_rows
                    if r["decided_by"] == "choice" and is_correct_all(r) is False
                ]
                noul_confs = [r["confidence"] for r in arm_rows if r["decided_by"] == "noul"]
            else:
                correct_confs = [r["confidence"] for r in arm_rows if is_correct_all(r) is True]
                wrong_confs = [r["confidence"] for r in arm_rows if is_correct_all(r) is False]
                noul_confs = []
            color = C_LLM if a in LLM else C_JEV

            total_counts = None
            for ax in (upper, lower):
                correct_counts, _, _ = ax.hist(
                    correct_confs,
                    bins=bins,
                    color=color,
                    edgecolor="white",
                    label="correct",
                )
                wrong_counts, _, _ = ax.hist(
                    wrong_confs,
                    bins=bins,
                    bottom=correct_counts,
                    color=color,
                    edgecolor="white",
                    hatch="///",
                    label="wrong",
                )
                total_counts = correct_counts + wrong_counts
                if noul_confs:
                    noul_counts, _, _ = ax.hist(
                        noul_confs,
                        bins=bins,
                        bottom=total_counts,
                        facecolor="white",
                        edgecolor=C_JEV,
                        hatch="...",
                        linewidth=1,
                        label="decided by the Noul",
                    )
                    total_counts += noul_counts
                ax.grid(axis="y", color=GRID, lw=0.6)
                ax.set_axisbelow(True)
                ax.spines["right"].set_visible(False)

            upper.set_ylim(65, 105)
            upper.set_yticks([70, 80, 90, 100])
            upper.spines["bottom"].set_visible(False)
            upper.tick_params(axis="x", bottom=False, labelbottom=False)
            panel_title = title.replace(", distribution", ",\ndistribution")
            upper.set_title(panel_title, fontsize=10 if "\n" in panel_title else 11, loc="left")

            lower.set_ylim(0, 20)
            lower.set_yticks([0, 5, 10, 15, 20])
            lower.spines["top"].set_visible(False)
            lower.set_xlabel("confidence")

            # Diagonal marks make the omitted 20–65 range explicit.
            d = 0.012
            break_style = {"color": INK, "clip_on": False, "lw": 1}
            upper.plot((-d, d), (-d, d), transform=upper.transAxes, **break_style)
            upper.plot((1 - d, 1 + d), (-d, d), transform=upper.transAxes, **break_style)
            lower.plot((-d, d), (1 - d, 1 + d), transform=lower.transAxes, **break_style)
            lower.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=lower.transAxes, **break_style)

            tallest = max(range(len(total_counts)), key=total_counts.__getitem__)
            upper.text(
                (bins[tallest] + bins[tallest + 1]) / 2,
                total_counts[tallest] + 1,
                f"{int(total_counts[tallest])}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        fig.supylabel("questions", x=0.025)
        fig.suptitle(
            "Representative confidence distributions, 106 questions each",
            fontsize=13,
            x=0.5,
            ha="center",
        )
        fig.legend(
            handles=[
                Patch(facecolor="#777777", edgecolor="white", label="correct"),
                Patch(facecolor="#777777", edgecolor="white", hatch="///", label="wrong"),
                Patch(
                    facecolor="white",
                    edgecolor="#777777",
                    hatch="...",
                    label="decided by the Noul",
                ),
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.88),
            ncol=3,
            frameon=False,
        )
        fig.text(
            0.5,
            0.02,
            textwrap.fill(
                "Confidence semantics differ; these distributions are descriptive and are not "
                "directly comparable as calibrated probabilities. In the Jev panel, the 3 dotted "
                "rows were decided by the yes/no Noul and plot its probability, not a Choice "
                "confidence.",
                width=135,
            ),
            fontsize=9,
            color="#555",
            ha="center",
        )
        fig.subplots_adjust(left=0.08, right=0.99, bottom=0.18, top=0.79, wspace=0.10)
        fig.savefig(outfile, dpi=200)
        return

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, a, t in zip(axes, [p[0] for p in panels], [p[1] for p in panels], strict=True):
        confs = [r["confidence"] for r in rows if r["classifier"] == a and r["confidence"] is not None]
        ax.hist(confs, bins=bins, color=C_LLM if a in LLM else C_JEV, edgecolor="white")
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
    fig.savefig(outfile, dpi=200)


# --- repo chart (3 panels, split_correct off) ---
PANELS = [
    ("luna", "Luna — self-reported"),
    ("sonnet", "Sonnet 5 — self-reported"),
    ("jev", "Bare Jev — Choice-distribution-derived"),
]
plot_confidence(
    PANELS,
    os.path.join(OUTDIR, "04-confidence.png"),
    split_correct=False,
)

# --- article chart (3 panels, split_correct on) ---
ARTICLE_PANELS = [
    ("luna", "luna, self-reported"),
    ("sonnet", "sonnet, self-reported"),
    ("jev_parallel_fewshot", "jev_parallel_fewshot, distribution-derived (+ Noul)"),
]
plot_confidence(
    ARTICLE_PANELS,
    os.path.join(ARTICLE_OUTDIR, "04-confidence.png"),
    split_correct=True,
)


# ── Chart 5 — shape × examples ───────────────────────────────────────────
def plot_shape_x_examples(outfile):
    categories = ["ask once", "ask in sequence", "ask together"]
    no_examples = [ACC[a] for a in ("jev", "jev_atomic", "jev_parallel")]
    with_examples = [
        ACC[a] for a in ("jev_fewshot", "jev_atomic_fewshot", "jev_parallel_fewshot")
    ]
    assert no_examples == [45, 45, 44], f"no examples: expected [45, 45, 44], got {no_examples}"
    assert with_examples == [46, 48, 50], (
        f"with examples: expected [46, 48, 50], got {with_examples}"
    )

    fig, ax = plt.subplots(figsize=(10, 5.4))
    x = list(range(len(categories)))
    ax.plot(x, no_examples, color="#777777", marker="o", lw=2, ms=7, label="no examples")
    ax.plot(x, with_examples, color=C_JEV, marker="o", lw=2, ms=7, label="with examples")
    for i, value in enumerate(no_examples):
        ax.annotate(
            str(value),
            (i, value),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            color="#666666",
            fontsize=9,
        )
    for i, value in enumerate(with_examples):
        ax.annotate(
            str(value),
            (i, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            color=C_JEV,
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(42, 52)
    ax.set_yticks([42, 44, 46, 48, 50, 52])
    ax.set_ylabel("correct, hard set (of 58)")
    ax.set_title("Shape and examples only paid off together", loc="left", fontsize=13, pad=14)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    footer = (
        "Run 3 ledger. Ask once = one nine-label Choice; in sequence = Noul, then an "
        "eight-label Choice; together = Noul and nine-label Choice in one request."
    )
    fig.text(0.5, 0.02, textwrap.fill(footer, width=115), fontsize=9, color="#555", ha="center")
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    fig.savefig(outfile, dpi=200)


plot_shape_x_examples(os.path.join(ARTICLE_OUTDIR, "05-shape-x-examples.png"))


# ── Chart 6 — Noul threshold sweep ───────────────────────────────────────
def plot_threshold_sweep(outfile):
    thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    arms = ["jev_parallel", "jev_parallel_fewshot"]
    deterministic_ids = {
        qid for qid, question in hard_questions.items() if question.get("caught_by") == "deterministic"
    }
    as_run = {}
    sweep = {}
    for arm in arms:
        arm_rows = [r for r in rows if r["classifier"] == arm]
        assert len(arm_rows) == 106, f"{arm}: expected 106 rows, got {len(arm_rows)}"
        as_run[arm] = sum(correct_any(r) for r in arm_rows)
        sweep[arm] = []
        for threshold in thresholds:
            total = 0
            for row in arm_rows:
                if row["question_id"] in deterministic_ids:
                    rescored_label = row["label"]
                elif row["noul_probability"] >= threshold:
                    rescored_label = "ambiguous"
                else:
                    rescored_label = row["choice_label"]
                rescored = dict(row, label=rescored_label)
                total += correct_any(rescored)
            sweep[arm].append(total)

    expected = {
        "jev_parallel": (90, [87, 91, 92, 92, 92, 91]),
        "jev_parallel_fewshot": (97, [95, 96, 95, 96, 96, 94]),
    }
    for arm, (expected_as_run, expected_sweep) in expected.items():
        assert as_run[arm] == expected_as_run, (
            f"{arm} as run: expected {expected_as_run}, got {as_run[arm]}"
        )
        assert sweep[arm] == expected_sweep, (
            f"{arm} sweep: expected {expected_sweep}, got {sweep[arm]}"
        )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    styles = {
        "jev_parallel": ("#777777", "o", -15),
        "jev_parallel_fewshot": (C_JEV, "s", 8),
    }
    for arm in arms:
        color, marker, label_offset = styles[arm]
        ax.plot(thresholds, sweep[arm], color=color, marker=marker, lw=2, ms=7, label=arm)
        for threshold, value in zip(thresholds, sweep[arm], strict=True):
            ax.annotate(
                str(value),
                (threshold, value),
                xytext=(0, label_offset),
                textcoords="offset points",
                ha="center",
                color=color,
                fontsize=8.5,
            )
        ax.axhline(
            as_run[arm],
            color=color,
            ls="--",
            lw=1.2,
            alpha=0.8,
            label=f"{arm} as run (Noul ≥ Choice confidence)",
        )
        ax.annotate(
            str(as_run[arm]),
            (thresholds[-1] + 0.006, as_run[arm]),
            xytext=(0, -13 if arm == "jev_parallel" else 7),
            textcoords="offset points",
            color=color,
            ha="center",
            va="center",
            fontsize=8.5,
        )
    ax.set_xticks(thresholds)
    ax.set_xticklabels([f"{threshold:.2f}" for threshold in thresholds])
    ax.set_xlim(0.44, 0.72)
    ax.set_ylim(85, 99)
    ax.set_xlabel("fixed Noul threshold")
    ax.set_ylabel("correct (of 106)")
    ax.set_title("Same sweep, opposite verdicts", loc="left", fontsize=13, pad=14)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    footer = (
        "Run 3 ledger, re-scored offline from stored Noul probabilities and Choice labels. "
        "No new calls. Thresholds chosen after seeing the data."
    )
    fig.text(0.5, 0.02, textwrap.fill(footer, width=115), fontsize=9, color="#555", ha="center")
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    fig.savefig(outfile, dpi=200)


plot_threshold_sweep(os.path.join(ARTICLE_OUTDIR, "06-threshold-sweep.png"))


# ── Chart 7 — cost per million classifications ───────────────────────────
def plot_cost_per_million(outfile):
    arms = ["jev", "jev_parallel_fewshot", "luna", "sonnet", "gpt56"]
    list_cost = {
        arm: sum(r["list_price_cost_usd"] for r in rows if r["classifier"] == arm)
        / 106
        * 1_000_000
        for arm in arms
    }
    cached_cost = {
        arm: sum(r["estimated_billed_cost_usd"] for r in rows if r["classifier"] == arm)
        / 106
        * 1_000_000
        for arm in ("luna", "gpt56")
    }
    expected_list = {
        "jev": 23.89,
        "jev_parallel_fewshot": 115.12,
        "luna": 407.39,
        "sonnet": 6448.79,
        "gpt56": 8305.58,
    }
    expected_cached = {"luna": 93.33, "gpt56": 2024.54}
    for arm, expected_value in expected_list.items():
        assert abs(list_cost[arm] - expected_value) <= 0.01, (
            f"{arm} list: expected {expected_value:.2f}, got {list_cost[arm]:.2f}"
        )
    for arm, expected_value in expected_cached.items():
        assert abs(cached_cost[arm] - expected_value) <= 0.01, (
            f"{arm} cached: expected {expected_value:.2f}, got {cached_cost[arm]:.2f}"
        )

    ordered = sorted(arms, key=list_cost.get)
    fig, ax = plt.subplots(figsize=(10, 5.7))
    bar_height = 0.28
    for y, arm in enumerate(ordered):
        color = C_JEV if arm.startswith("jev") else C_LLM
        has_cached = arm in cached_cost
        list_y = y - bar_height / 1.6 if has_cached else y
        ax.barh(list_y, list_cost[arm], height=bar_height, color=color)
        ax.text(
            list_cost[arm] * 1.06,
            list_y,
            f"${list_cost[arm]:,.0f}",
            va="center",
            fontsize=9,
            color=INK,
        )
        if has_cached:
            cached_y = y + bar_height / 1.6
            ax.barh(
                cached_y,
                cached_cost[arm],
                height=bar_height,
                color=color,
                edgecolor="white",
                hatch="///",
            )
            ax.text(
                cached_cost[arm] * 1.06,
                cached_y,
                f"${cached_cost[arm]:,.0f}",
                va="center",
                fontsize=9,
                color=INK,
            )
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(10, 14_000)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    ax.set_xlabel("cost per million classifications (USD, log scale)")
    ax.set_title("What a million routed questions would cost", loc="left", fontsize=13, pad=14)
    ax.legend(
        handles=[
            Patch(facecolor="#777777", label="list price"),
            Patch(facecolor="#777777", edgecolor="white", hatch="///", label="est. with caching"),
        ],
        frameon=False,
        loc="upper right",
    )
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    footer = (
        "Linear extrapolation of one 106-question run. Cached figures are estimates rebuilt "
        "from reported cached-token counts. Sonnet ran without caching; Jev's API reports no "
        "cached tokens."
    )
    fig.text(0.5, 0.02, textwrap.fill(footer, width=115), fontsize=9, color="#555", ha="center")
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(outfile, dpi=200)


plot_cost_per_million(os.path.join(ARTICLE_OUTDIR, "07-cost-per-million.png"))


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
