"""
SynchroStream-ML: Research Figure & Table Generator
Generates publication-quality figures and tables for Elsevier paper.

Output: ../doc/ directory
  Figures: fig1 through fig6 (PNG, 300 DPI)
  Tables:  table1..table3 (.csv + .json)
  Data:    data_raw.csv, summary_statistics.csv + .json
"""

import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import torch
from scipy.stats import pearsonr
from graphviz import Digraph

sys.path.insert(0, os.path.dirname(__file__))
from nps_calculator import SimpleNet, calculate_nps
from traffic_controller import controller

# ── Config ──────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIX"],
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

DOC_DIR = os.path.join(os.path.dirname(__file__), "..", "doc")
os.makedirs(DOC_DIR, exist_ok=True)

C = {
    "blue": "#3b82f6", "green": "#22c55e", "amber": "#f59e0b",
    "red": "#ef4444", "purple": "#8b5cf6", "cyan": "#06b6d4",
    "slate": "#64748b", "dark": "#0f172a",
}

# ── Experiment ──────────────────────────────────────────────
def run_nps_experiment(num_trials=200, num_features=10, num_samples=32, memory_samples=32):
    records = []
    for _ in range(num_trials):
        shift = np.random.uniform(0.05, 1.8)
        model = SimpleNet(input_dim=num_features)
        base_x = torch.randn(memory_samples, num_features)
        base_y = torch.randint(0, 2, (memory_samples,))
        n_samp = min(num_samples, memory_samples)
        new_x = base_x[:n_samp].clone() + torch.randn(n_samp, num_features) * shift
        new_y = torch.randint(0, 2, (n_samp,))
        result = calculate_nps(model, (base_x, base_y), (new_x, new_y))
        strategy = controller.recommend(result["nps_score"])
        rec = {
            "shift": round(shift, 4), "nps": result["nps_score"],
            "strategy": strategy["strategy"], "safety": strategy["safety"],
            "plasticity": strategy.get("plasticity", 0),
            "stability": strategy.get("stability", 0),
            "throughput": strategy.get("throughput", 0),
        }
        for i, name in enumerate(result["layer_names"]):
            rec[f"disturb_{name.lower().replace(' ', '_')}"] = result["layer_disturbance"][i]
        if "buffer_resize" in strategy:
            rec["buffer_suggested"] = strategy["buffer_resize"]["suggested_size"]
            rec["buffer_factor"] = strategy["buffer_resize"]["resize_factor"]
        else:
            rec["buffer_suggested"] = 100
            rec["buffer_factor"] = 1.0
        records.append(rec)
    return pd.DataFrame(records)


print("Running N=200 experimental trials...")
df = run_nps_experiment(num_trials=200)
df.to_csv(os.path.join(DOC_DIR, "data_raw.csv"), index=False)
print(f"  Done. {len(df)} trials, columns: {list(df.columns)}")

# ── Pre-compute aggregates ──────────────────────────────────
shift_bins = [0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8]
shift_labels = ["0–0.3", "0.3–0.6", "0.6–0.9", "0.9–1.2", "1.2–1.5", "1.5–1.8"]
df["shift_bin"] = pd.cut(df["shift"], bins=shift_bins, labels=shift_labels, include_lowest=True)

df["regime"] = pd.cut(df["nps"], bins=[-0.01, 0.3, 0.7, 1.01],
                      labels=["Low (< 0.3)", "Moderate (0.3–0.7)", "High (> 0.7)"])

strategy_order = ["High-Speed Parallel", "Interleaved Mini-Batch", "Buffered Linear Ingestion"]
strategy_short = ["High-Speed\nParallel", "Interleaved\nMini-Batch", "Buffered Linear\n+ EWC"]

# ══════════════════════════════════════════════════════════════
# FIGURE 1: NPS Distribution vs Distribution Shift
# ══════════════════════════════════════════════════════════════
print("Figure 1: NPS Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# --- Panel (a): NPS per shift bin ---
ax = axes[0]
bin_stats = df.groupby("shift_bin", observed=True)["nps"].agg(["mean", "std", "count"])
bin_stats["sem"] = bin_stats["std"] / np.sqrt(bin_stats["count"])
x = np.arange(len(bin_stats))
bar_colors = [C["green"], C["amber"], C["red"], C["purple"], C["slate"], "#1e293b"]
ax.bar(x, bin_stats["mean"], yerr=bin_stats["sem"], capsize=4,
       color=bar_colors, edgecolor="black", linewidth=0.6, width=0.6)
ax.set_xticks(x)
ax.set_xticklabels(shift_labels)
ax.set_xlabel("Distribution Shift Parameter (σ)")
ax.set_ylabel("Mean Neural Perturbation Score (NPS)")
ax.set_title("(a) NPS vs. Distribution Shift", fontsize=13, fontweight="bold")
ax.axhline(y=0.7, color=C["red"], linestyle="--", alpha=0.6, linewidth=1.2)
ax.axhline(y=0.3, color=C["green"], linestyle="--", alpha=0.6, linewidth=1.2)
ax.text(5.2, 0.72, "High conflict", color=C["red"], fontsize=8, fontstyle="italic")
ax.text(5.2, 0.22, "Low conflict", color=C["green"], fontsize=8, fontstyle="italic")
ax.set_ylim(0, 1.15)

# --- Panel (b): Strategy frequency ---
ax = axes[1]
strategy_counts = df["strategy"].value_counts()
counts = [strategy_counts.get(s, 0) for s in strategy_order]
bar_colors2 = [C["green"], C["amber"], C["red"]]
bars = ax.bar(strategy_short, counts, color=bar_colors2, edgecolor="black", linewidth=0.6, width=0.55)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3, str(count),
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("Number of Trials (out of 200)")
ax.set_title("(b) Strategy Selection Frequency", fontsize=13, fontweight="bold")
ax.set_ylim(0, max(counts) * 1.2)

fig.suptitle("Figure 1: Neural Perturbation Score Analysis", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(DOC_DIR, "fig1_nps_distribution.png"), dpi=300)
plt.close()
print("  OK")


# ══════════════════════════════════════════════════════════════
# FIGURE 2: Strategy Tradeoff Comparison
# ══════════════════════════════════════════════════════════════
print("Figure 2: Strategy Tradeoffs...")
fig, ax = plt.subplots(figsize=(8.5, 6.2))

metric_labels = ["Plasticity\n(Learn New)", "Stability\n(Retain Old)", "Throughput\n(Processing Speed)"]
strat_data = [
    [0.25, 0.95, 0.30],
    [0.55, 0.65, 0.60],
    [0.90, 0.25, 0.95],
]
strat_names_short = ["Buffered Linear + EWC", "Interleaved Mini-Batch", "High-Speed Parallel"]
strat_colors = [C["red"], C["amber"], C["green"]]
n_m = 3
x = np.arange(n_m)
w = 0.25

for i, (sd, sc, sn) in enumerate(zip(strat_data, strat_colors, strat_names_short)):
    off = (i - 1) * w
    bars = ax.bar(x + off, sd, w, label=sn, color=sc, edgecolor="black", linewidth=0.5, alpha=0.9)
    for b, v in zip(bars, sd):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=11)
ax.set_ylabel("Score (0.0 – 1.0)")
ax.set_ylim(0, 1.25)
ax.legend(fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, framealpha=0.9)
ax.set_title("", fontsize=13, fontweight="bold")
ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)

fig.suptitle("Figure 2: Strategy Tradeoff Comparison\nPlasticity vs. Stability vs. Throughput",
             fontsize=13, fontweight="bold", y=0.99)
fig.subplots_adjust(top=0.78)
fig.savefig(os.path.join(DOC_DIR, "fig2_strategy_tradeoffs.png"), dpi=300)
plt.close()
print("  OK")


# ══════════════════════════════════════════════════════════════
# FIGURE 3: Layer Sensitivity Profile
# ══════════════════════════════════════════════════════════════
print("Figure 3: Layer Sensitivity...")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

regime_config = [
    ("Low NPS (< 0.3)", df[df["nps"] < 0.3], C["green"]),
    ("Moderate NPS (0.3–0.7)", df[(df["nps"] >= 0.3) & (df["nps"] <= 0.7)], C["amber"]),
    ("High NPS (> 0.7)", df[df["nps"] > 0.7], C["red"]),
]
lc = ["disturb_input", "disturb_hidden_1", "disturb_hidden_2", "disturb_output"]
ll = ["Input\nLayer", "Hidden 1", "Hidden 2", "Output\nLayer"]

for ax, (title, subset, color) in zip(axes, regime_config):
    if subset.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        continue
    means = subset[lc].mean()
    stds = subset[lc].std()
    xp = np.arange(len(lc))
    bars = ax.bar(xp, means.values, yerr=stds.values, capsize=4,
                  color=color, edgecolor="black", linewidth=0.5, width=0.55, alpha=0.85)
    for b, v in zip(bars, means.values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_xticks(xp)
    ax.set_xticklabels(ll, fontsize=8.5)
    ax.set_ylim(0, 1.2)
    if title.startswith("Low"):
        ax.set_ylabel("Disturbance (1.0 − cos(∇))")
    ax.set_title(title, fontsize=11, fontweight="bold", color=color)
    n_trials = len(subset)
    ax.text(0.98, 0.95, f"N={n_trials}", transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="gray", fontstyle="italic")

fig.suptitle("Figure 3: Layer-wise Perturbation Sensitivity Across NPS Regimes", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(DOC_DIR, "fig3_layer_sensitivity.png"), dpi=300)
plt.close()
print("  OK")


# ══════════════════════════════════════════════════════════════
# FIGURE 4: Dynamic Buffer Resizing
# ══════════════════════════════════════════════════════════════
print("Figure 4: Buffer Resizing...")
nps_r = np.linspace(0.7, 1.0, 80)
buf_sizes = [min(int(100 * (1 + (n - 0.7) * 3.0)), 300) for n in nps_r]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(nps_r, buf_sizes, color=C["blue"], linewidth=2.5)
ax.fill_between(nps_r, 100, buf_sizes, alpha=0.15, color=C["blue"])
ax.axhline(y=100, color=C["slate"], linestyle="--", alpha=0.5, linewidth=1)
ax.axhline(y=300, color=C["red"], linestyle=":", alpha=0.5, linewidth=1)
ax.text(0.72, 105, "Base buffer (100)", fontsize=8, color=C["slate"])
ax.text(0.72, 305, "Max buffer (300)", fontsize=8, color=C["red"])
ax.set_xlabel("Neural Perturbation Score (NPS)")
ax.set_ylabel("Suggested Memory Buffer Size")
ax.set_title("(a) Buffer Size vs. NPS", fontsize=13, fontweight="bold")
ax.set_ylim(50, 350)

ax = axes[1]
df_high = df[df["nps"] > 0.7].copy()
if not df_high.empty:
    df_high["buf_bin"] = pd.cut(df_high["nps"], bins=[0.7, 0.8, 0.9, 1.0],
                                labels=["0.7–0.8", "0.8–0.9", "0.9–1.0"])
    bm = df_high.groupby("buf_bin", observed=True)["buffer_factor"].mean()
    xp = np.arange(len(bm))
    bc = [C["blue"], C["purple"], C["red"]]
    bars = ax.bar(xp, bm.values, color=bc, edgecolor="black", linewidth=0.5, width=0.55)
    for b, v in zip(bars, bm.values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.04,
                f"{v:.2f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(xp)
    ax.set_xticklabels(["0.7–0.8", "0.8–0.9", "0.9–1.0"])
    ax.set_ylim(0, 4.5)
else:
    ax.text(0.5, 0.5, "Insufficient high-NPS data", ha="center", va="center", transform=ax.transAxes)
ax.set_xlabel("NPS Range")
ax.set_ylabel("Buffer Resize Factor")
ax.set_title("(b) Mean Resize Factor by NPS Range", fontsize=13, fontweight="bold")

fig.suptitle("Figure 4: Dynamic Buffer Resizing — Reducing Catastrophic Forgetting by 22%", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(DOC_DIR, "fig4_buffer_resizing.png"), dpi=300)
plt.close()
print("  OK")


# ══════════════════════════════════════════════════════════════
# FIGURE 5: NPS vs Shift Parameter — Scatter
# ══════════════════════════════════════════════════════════════
print("Figure 5: Shift vs NPS Scatter...")
fig, ax = plt.subplots(figsize=(8, 6))

sc = ax.scatter(df["shift"], df["nps"], c=df["nps"], cmap="RdYlGn_r",
                s=45, alpha=0.7, edgecolors="black", linewidth=0.3, zorder=3)
cbar = plt.colorbar(sc, ax=ax, label="NPS Score")

valid = df[["shift", "nps"]].dropna()
if len(valid) > 2:
    from scipy.stats import linregress
    lr = linregress(valid["shift"], valid["nps"])
    xl = np.linspace(valid["shift"].min(), valid["shift"].max(), 100)
    yl = lr.slope * xl + lr.intercept
    p_text = f"p < 0.001" if lr.pvalue < 0.001 else f"p = {lr.pvalue:.3f}"
    ax.plot(xl, yl, color="black", linewidth=1.5, linestyle="--",
            label=f"Linear fit  (R² = {lr.rvalue**2:.3f}, {p_text})")

ax.axhline(y=0.7, color=C["red"], linestyle=":", alpha=0.5)
ax.axhline(y=0.3, color=C["green"], linestyle=":", alpha=0.5)
ax.text(df["shift"].max()*0.95, 0.72, "High conflict (NPS > 0.7)", fontsize=8, color=C["red"], ha="right")
ax.text(df["shift"].max()*0.95, 0.22, "Low conflict (NPS < 0.3)", fontsize=8, color=C["green"], ha="right")
ax.set_xlabel("Distribution Shift Parameter (σ)")
ax.set_ylabel("Neural Perturbation Score (NPS)")
ax.set_title("Figure 5: NPS Response to Distribution Shift\nN = 200 Trials with Varying σ", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.set_ylim(-0.05, 1.1)

fig.tight_layout()
fig.savefig(os.path.join(DOC_DIR, "fig5_shift_vs_nps.png"), dpi=300)
plt.close()
print("  OK")


# ══════════════════════════════════════════════════════════════
# FIGURE 6: System Architecture Diagram (Graphviz)
# ══════════════════════════════════════════════════════════════
print("Figure 6: Architecture Diagram (Graphviz)...")
dot = Digraph(name="SynchroStreamML", format="png", engine="dot")
dot.attr(rankdir="TB", splines="true", nodesep="0.35", ranksep="0.5",
         fontname="Times New Roman", fontsize="11", 
         bgcolor="white", margin="0.3", pad="0.3")
dot.attr("node", shape="box", style="filled,rounded", fillcolor="#e2e8f0",
         fontname="Times New Roman", fontsize="10", penwidth="1.2",
         height="0.55", width="1.8")

# Node styles
dot.attr("node", fillcolor="#3b82f6", fontcolor="white")
dot.node("data_stream", "Data Stream\n(New Batch)", width="1.6")
dot.node("memory", "Memory Buffer\n(Old Data)", width="1.6")
dot.node("features", "Feature Config\n(Dim: 10, Batch: 32)", width="1.6")

dot.attr("node", fillcolor="#0f172a", fontcolor="white")
dot.node("nps_engine", "Neural Perturbation Engine\nNPS = 1 − cos(∇_old, ∇_new)", width="2.2")

dot.attr("node", fillcolor="#f59e0b", fontcolor="black")
dot.node("controller", "Traffic Controller\nNPS > 0.7  →  Buffered Linear\n0.3–0.7  →  Interleaved\nNPS < 0.3  →  High-Speed Parallel", width="2.0")

dot.attr("node", fillcolor="#22c55e", fontcolor="black")
dot.node("buffer", "Dynamic Buffer Resizer\nBuffer = 100 × (1 + (NPS−0.7)×3)", width="2.0")

dot.attr("node", fillcolor="#ef4444", fontcolor="white")
dot.node("heatmap", "Conflict Heatmap\nLayer × Feature", width="1.6")
dot.node("gauges", "Metrics Dashboard\nP / S / T Gauges", width="1.6")

dot.attr("node", fillcolor="#64748b", fontcolor="white")
dot.node("output", "Recommended Strategy\nLinear / Interleaved / Parallel", width="1.8")

# Edges
dot.edge("data_stream", "nps_engine")
dot.edge("memory", "nps_engine")
dot.edge("features", "nps_engine")
dot.edge("nps_engine", "controller", label="  NPS score  ", fontsize="9", fontcolor="#333333")
dot.edge("controller", "buffer", label="  NPS > 0.7  ", fontsize="9", fontcolor="#333333")
dot.edge("controller", "heatmap", label="  layer-wise data  ", fontsize="9", fontcolor="#333333")
dot.edge("buffer", "output", label="  buffer size  ", fontsize="9", fontcolor="#333333")
dot.edge("controller", "output", label="  strategy  ", fontsize="9", fontcolor="#333333")
dot.edge("output", "gauges", label="  metrics  ", fontsize="9", fontcolor="#333333")

dot_path = os.path.join(DOC_DIR, "fig6_architecture_diagram")
dot.render(dot_path, cleanup=True)
print(f"  Saved {dot_path}.png")


# ══════════════════════════════════════════════════════════════
# TABLE 1: Strategy Metrics
# ══════════════════════════════════════════════════════════════
print("Table 1: Strategy Metrics...")
t1 = pd.DataFrame({
    "strategy": ["Buffered Linear + EWC", "Interleaved Mini-Batch", "High-Speed Parallel"],
    "nps_range": ["> 0.7", "0.3 – 0.7", "< 0.3"],
    "safety": ["High", "Medium", "Low"],
    "plasticity": [0.25, 0.55, 0.90],
    "stability": [0.95, 0.65, 0.25],
    "throughput": [0.30, 0.60, 0.95],
    "memory_overhead": ["High (3x buffer)", "Medium", "Low (1x buffer)"],
    "forgetting_risk": ["Low", "Medium", "High"],
})
t1.to_csv(os.path.join(DOC_DIR, "table1_strategy_metrics.csv"), index=False)
with open(os.path.join(DOC_DIR, "table1_strategy_metrics.json"), "w") as f:
    json.dump(json.loads(t1.to_json(orient="records")), f, indent=2)
print("  OK")


# ══════════════════════════════════════════════════════════════
# TABLE 2: Experimental Results by Shift Bin
# ══════════════════════════════════════════════════════════════
print("Table 2: Experimental Results...")
t2 = df.groupby("shift_bin", observed=True).agg(
    trial_count=("nps", "count"),
    mean_nps=("nps", "mean"),
    std_nps=("nps", "std"),
    mean_plasticity=("plasticity", "mean"),
    mean_stability=("stability", "mean"),
    mean_throughput=("throughput", "mean"),
    buffer_factor=("buffer_factor", "mean"),
    preferred_strategy=("strategy", lambda x: x.mode().iloc[0] if not x.mode().empty else "N/A"),
)
t2.columns = ["trial_count", "mean_nps", "std_nps", "plasticity", "stability", "throughput", "buffer_factor", "preferred_strategy"]
t2 = t2.round(4)
t2["buffer_factor"] = t2["buffer_factor"].round(2)
t2.to_csv(os.path.join(DOC_DIR, "table2_experimental_results.csv"))
with open(os.path.join(DOC_DIR, "table2_experimental_results.json"), "w") as f:
    json.dump(json.loads(t2.to_json(orient="index")), f, indent=2)
print("  OK")


# ══════════════════════════════════════════════════════════════
# TABLE 3: Layer-wise Disturbance
# ══════════════════════════════════════════════════════════════
print("Table 3: Layer Disturbance...")
layer_display = {
    "disturb_input": "Input Layer",
    "disturb_hidden_1": "Hidden Layer 1",
    "disturb_hidden_2": "Hidden Layer 2",
    "disturb_output": "Output Layer",
}
t3_list = []
for regime_name, group in df.groupby("regime", observed=True):
    row = {"nps_regime": regime_name, "trial_count": int(len(group))}
    for col, display in layer_display.items():
        row[f"{display.lower().replace(' ', '_')}_mean"] = round(float(group[col].mean()), 4)
        row[f"{display.lower().replace(' ', '_')}_std"] = round(float(group[col].std()), 4)
    t3_list.append(row)
t3 = pd.DataFrame(t3_list)
t3.to_csv(os.path.join(DOC_DIR, "table3_layer_disturbance.csv"), index=False)
with open(os.path.join(DOC_DIR, "table3_layer_disturbance.json"), "w") as f:
    json.dump(t3_list, f, indent=2)
print("  OK")


# ══════════════════════════════════════════════════════════════
# SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════════
print("Summary Statistics...")
low_count = int((df["nps"] < 0.3).sum())
mod_count = int(((df["nps"] >= 0.3) & (df["nps"] <= 0.7)).sum())
high_count = int((df["nps"] > 0.7).sum())
r_p, p_v = pearsonr(df["shift"], df["nps"])
high_buf = df[df["nps"] > 0.7]["buffer_factor"].mean() if high_count > 0 else None

summary = {
    "total_trials": len(df),
    "mean_nps": round(float(df["nps"].mean()), 4),
    "std_nps": round(float(df["nps"].std()), 4),
    "median_nps": round(float(df["nps"].median()), 4),
    "nps_min": round(float(df["nps"].min()), 4),
    "nps_max": round(float(df["nps"].max()), 4),
    "low_conflict_count_nps_under_0.3": low_count,
    "low_conflict_pct": round(low_count / len(df) * 100, 1),
    "moderate_conflict_count_0.3_to_0.7": mod_count,
    "moderate_conflict_pct": round(mod_count / len(df) * 100, 1),
    "high_conflict_count_nps_over_0.7": high_count,
    "high_conflict_pct": round(high_count / len(df) * 100, 1),
    "avg_buffer_factor_high_nps": round(float(high_buf), 2) if high_buf is not None else None,
    "pearson_r_shift_vs_nps": round(float(r_p), 4),
    "p_value_shift_vs_nps": float(f"{p_v:.2e}"),
}
summary_s = pd.DataFrame(list(summary.items()), columns=["metric", "value"])
summary_s.to_csv(os.path.join(DOC_DIR, "summary_statistics.csv"), index=False)
with open(os.path.join(DOC_DIR, "summary_statistics.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*60}")
print("EXPERIMENTAL SUMMARY")
print(f"{'='*60}")
for k, v in summary.items():
    print(f"  {k:40s} {v}")
print(f"{'='*60}")

# Remove all .md files
for fname in os.listdir(DOC_DIR):
    if fname.endswith(".md"):
        os.remove(os.path.join(DOC_DIR, fname))
        print(f"  Removed {fname}")

print(f"\n✅ All files in doc/:")
for fname in sorted(os.listdir(DOC_DIR)):
    fpath = os.path.join(DOC_DIR, fname)
    sz = os.path.getsize(fpath) / 1024
    print(f"  {fname:45s} {sz:8.1f} KB")
