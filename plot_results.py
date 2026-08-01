"""
Reads saved result CSVs from the results/ folder and generates
a cumulative power consumption plot.

Each policy run saves a CSV with at minimum:
    actual_power_W    — network power consumed at each timestep
    baseline_power_W  — always-on power at each timestep
    timestep          — step index

Usage:
    python plot_results.py

Looks for these files in results/ (eval split):
    always_on_eval.csv
    threshold_eval.csv
    cql_eval.csv        ← skipped if not found
    iql_eval.csv        ← skipped if not found
    dqn_eval.csv        ← skipped if not found

Output:
    results/cumulative_power.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Config
RESULTS_DIR = "results"
OUTPUT_PATH = os.path.join(RESULTS_DIR, "cumulative_power.png")

# Policy display config — add RL entries here when ready, nothing else changes
POLICIES = [
    {
        "file":    "always_on_eval.csv",
        "label":   "Always-On (Baseline)",
        "color":   "#e74c3c",
        "ls":      "-",
        "required": True,
    },
    {
        "file":    "threshold_eval.csv",
        "label":   "Threshold Policy",
        "color":   "#2ecc71",
        "ls":      "-",
        "required": True,
    },
    # RL policies added here later
    {
        "file":    "cql_eval.csv",
        "label":   "CQL",
        "color":   "#3498db",
        "ls":      "-",
        "required": False,   # skipped if file not found
    },
    {
        "file":    "dqn_eval.csv",
        "label":   "DQN (Baseline RL)",
        "color":   "#f39c12",
        "ls":      "--",
        "required": False,
    },
    {
        "file":    "iql_eval.csv",
        "label":   "IQL",
        "color":   "#9b59b6",
        "ls":      "-.",
        "required": False,
    },
]


# Plot
def plot_cumulative_power():
    fig, ax = plt.subplots(figsize=(12, 6))

    plotted = []

    for p in POLICIES:
        path = os.path.join(RESULTS_DIR, p["file"])

        if not os.path.exists(path):
            if p["required"]:
                raise FileNotFoundError(
                    f"Required results file not found: {path}\n"
                    f"Run: python main.py --policy "
                    f"{p['file'].replace('_eval.csv','')} --split eval"
                )
            else:
                print(f"Skipping {p['label']} — file not found: {path}")
                continue

        df  = pd.read_csv(path)
        cum = df["actual_power_W"].cumsum().values
        x   = np.arange(len(cum))

        ax.plot(
            x, cum,
            label=p["label"],
            color=p["color"],
            linestyle=p["ls"],
            linewidth=2.0,
        )
        plotted.append(p["label"])

    # Formatting
    ax.set_xlabel("Timestep (hours)", fontsize=13)
    ax.set_ylabel("Cumulative Power Consumed (W·h)", fontsize=13)
    ax.set_title(
        "Cumulative Network Power Consumption — Evaluation Week\n"
        "(lower is better)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:,.0f}"
    ))

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nPlot saved → {OUTPUT_PATH}")

    # Summary table
    print(f"\n{'Policy':<25} {'Total Power (W·h)':>20} {'vs Always-On':>15}")
    print("-" * 62)

    baseline_total = None
    rows = []

    for p in POLICIES:
        path = os.path.join(RESULTS_DIR, p["file"])
        if not os.path.exists(path):
            continue
        df    = pd.read_csv(path)
        total = df["actual_power_W"].sum()
        rows.append((p["label"], total))
        if p["file"] == "always_on_eval.csv":
            baseline_total = total

    for label, total in rows:
        if baseline_total and label != "Always-On (Baseline)":
            saving = (baseline_total - total) / baseline_total * 100
            print(f"{label:<25} {total:>20,.2f} {saving:>14.1f}%")
        else:
            print(f"{label:<25} {total:>20,.2f} {'—':>15}")

def plot_scheme_comparison():
    """
    Grouped bar chart: % energy saved vs baseline, per policy, per scheme.
        Scheme 1 — Full model (macro compensation cost included)
        Scheme 2 — Micro-only savings (macros ignored entirely)
        Scheme 3 — Macros frozen at baseline load (no handoff cost)
    """
    labels = []
    scheme1, scheme2, scheme3 = [], [], []

    for p in POLICIES:
        path = os.path.join(RESULTS_DIR, p["file"])
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)

        base_full   = df["baseline_power_W"].mean()
        actual_full = df["actual_power_W"].mean()
        s1 = (1 - actual_full / base_full) * 100 if base_full > 0 else 0.0

        base_micro   = df["baseline_power_micro_only_W"].mean()
        actual_micro = df["actual_power_micro_only_W"].mean()
        s2 = (1 - actual_micro / base_micro) * 100 if base_micro > 0 else 0.0

        actual_macro_const = df["actual_power_macro_const_W"].mean()
        s3 = (1 - actual_macro_const / base_full) * 100 if base_full > 0 else 0.0

        labels.append(p["label"])
        scheme1.append(s1)
        scheme2.append(s2)
        scheme3.append(s3)

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, scheme1, width, label="Scheme 1: Full model (w/ macro cost)", color="#3498db")
    ax.bar(x,          scheme2, width, label="Scheme 2: Micro-only", color="#2ecc71")
    ax.bar(x + width,  scheme3, width, label="Scheme 3: Macros frozen (no handoff cost)", color="#e74c3c")

    ax.set_xlabel("Policy", fontsize=13)
    ax.set_ylabel("Energy Saved vs Baseline (%)", fontsize=13)
    ax.set_title(
        "Energy Savings by Power Accounting Scheme — Evaluation Week",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "scheme_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Scheme comparison plot saved → {out_path}")

    # Print table
    print(f"\n{'Policy':<25} {'Scheme 1 (%)':>14} {'Scheme 2 (%)':>14} {'Scheme 3 (%)':>14}")
    print("-" * 69)
    for label, s1, s2, s3 in zip(labels, scheme1, scheme2, scheme3):
        print(f"{label:<25} {s1:>13.1f}% {s2:>13.1f}% {s3:>13.1f}%")

if __name__ == "__main__":
    plot_cumulative_power()
    plot_scheme_comparison()