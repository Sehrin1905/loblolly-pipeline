# /// script
# requires-python = ">=3.13"
# dependencies = ["pandas", "matplotlib", "numpy"]
# ///
"""
Generate all biological figures from classified cell data.

Usage:
    uv run scripts/make_figures.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/all_samples.csv")

# Parse sample metadata from folder name
def parse_sample(name):
    name = name.replace("RETRY_", "").replace("***", "")
    parts = name.split("_")
    if parts[0].lower() in ("marsh", "far"):
        site = "Marsh"
    else:
        site = "Reference"
    period = parts[-1].capitalize()  # normalizes OLD/RECENT/Old/Recent
    return site, period

df[["site", "period"]] = df["sample"].apply(
    lambda x: pd.Series(parse_sample(x))
)

# Output folder
Path("data/figures").mkdir(exist_ok=True)

# ── Color palette ─────────────────────────────────────────────────────────────
MARSH_COLOR   = "#2E86AB"   # blue
REF_COLOR     = "#E84855"   # red
RECENT_COLOR  = "#3BB273"   # green
OLD_COLOR     = "#F18F01"   # orange

# ── Figure 1: Cell type proportions by sample ─────────────────────────────────
print("Making Figure 1: Cell type proportions by sample...")

summary = df.groupby(["sample", "cell_type"]).size().unstack(fill_value=0)
summary["total"] = summary.sum(axis=1)
summary["ray_pct"] = summary["ray"] / summary["total"] * 100
summary["tracheid_pct"] = summary["tracheid"] / summary["total"] * 100
summary = summary.sort_values("ray_pct", ascending=False)

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(summary))
ax.bar(x, summary["tracheid_pct"], label="Tracheid", color="#8B4513")
ax.bar(x, summary["ray_pct"], bottom=summary["tracheid_pct"], label="Ray", color="#90EE90")
ax.set_xticks(x)
ax.set_xticklabels(summary.index, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Cell type (%)")
ax.set_title("Cell Type Proportions by Sample")
ax.legend()
plt.tight_layout()
plt.savefig("data/figures/fig1_cell_type_proportions.png", dpi=150)
plt.close()
print("  Saved → data/figures/fig1_cell_type_proportions.png")

# ── Figure 2: Marsh vs Reference ray percentage ───────────────────────────────
print("Making Figure 2: Marsh vs Reference...")

site_summary = df.groupby(["sample", "site", "cell_type"]).size().unstack(fill_value=0).reset_index()
site_summary["total"] = site_summary[["ray", "tracheid"]].sum(axis=1)
site_summary["ray_pct"] = site_summary["ray"] / site_summary["total"] * 100

marsh = site_summary[site_summary["site"] == "Marsh"]["ray_pct"]
ref   = site_summary[site_summary["site"] == "Reference"]["ray_pct"]

fig, ax = plt.subplots(figsize=(6, 5))
ax.boxplot([marsh, ref], tick_labels=["Marsh", "Reference"], patch_artist=True,
           boxprops=dict(facecolor="lightblue"),
           medianprops=dict(color="black", linewidth=2))
ax.set_ylabel("Ray cells (%)")
ax.set_title("Ray Cell Proportion: Marsh vs Reference")
plt.tight_layout()
plt.savefig("data/figures/fig2_marsh_vs_reference.png", dpi=150)
plt.close()
print("  Saved → data/figures/fig2_marsh_vs_reference.png")

# ── Figure 3: Recent vs Old ray percentage ────────────────────────────────────
print("Making Figure 3: Recent vs Old...")

period_summary = df.groupby(["sample", "period", "cell_type"]).size().unstack(fill_value=0).reset_index()
period_summary["total"] = period_summary[["ray", "tracheid"]].sum(axis=1)
period_summary["ray_pct"] = period_summary["ray"] / period_summary["total"] * 100

recent = period_summary[period_summary["period"] == "Recent"]["ray_pct"]
old    = period_summary[period_summary["period"] == "Old"]["ray_pct"]

fig, ax = plt.subplots(figsize=(6, 5))
ax.boxplot([recent, old], tick_labels=["Recent", "Old"], patch_artist=True,
           boxprops=dict(facecolor="lightyellow"),
           medianprops=dict(color="black", linewidth=2))
ax.set_ylabel("Ray cells (%)")
ax.set_title("Ray Cell Proportion: Recent vs Old Wood")
plt.tight_layout()
plt.savefig("data/figures/fig3_recent_vs_old.png", dpi=150)
plt.close()
print("  Saved → data/figures/fig3_recent_vs_old.png")

# ── Figure 4: Total cells per sample ─────────────────────────────────────────
print("Making Figure 4: Total cells per sample...")

cell_counts = df.groupby("sample").size().sort_values(ascending=False)
colors = [MARSH_COLOR if "Marsh" in s or "Far" in s else REF_COLOR for s in cell_counts.index]

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(range(len(cell_counts)), cell_counts.values, color=colors)
ax.set_xticks(range(len(cell_counts)))
ax.set_xticklabels(cell_counts.index, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Total cells detected")
ax.set_title("Total Cells Detected per Sample")
marsh_patch = mpatches.Patch(color=MARSH_COLOR, label="Marsh")
ref_patch   = mpatches.Patch(color=REF_COLOR, label="Reference")
ax.legend(handles=[marsh_patch, ref_patch])
plt.tight_layout()
plt.savefig("data/figures/fig4_cell_counts.png", dpi=150)
plt.close()
print("  Saved → data/figures/fig4_cell_counts.png")

# ── Figure 5: Cell area distribution by site ──────────────────────────────────
print("Making Figure 5: Cell area distribution...")

marsh_areas = df[df["site"] == "Marsh"]["area"]
ref_areas   = df[df["site"] == "Reference"]["area"]

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(marsh_areas, bins=100, alpha=0.6, color=MARSH_COLOR, label="Marsh", density=True)
ax.hist(ref_areas,   bins=100, alpha=0.6, color=REF_COLOR,   label="Reference", density=True)
ax.set_xlabel("Cell lumen area (px²)")
ax.set_ylabel("Density")
ax.set_title("Cell Lumen Area Distribution: Marsh vs Reference")
ax.set_xlim(0, 2000)
ax.legend()
plt.tight_layout()
plt.savefig("data/figures/fig5_area_distribution.png", dpi=150)
plt.close()
print("  Saved → data/figures/fig5_area_distribution.png")

# ── Figure 6: Wall thickness by site and period ───────────────────────────────
print("Making Figure 6: Wall thickness...")

fig, ax = plt.subplots(figsize=(8, 5))
groups = [
    df[(df["site"] == "Marsh")     & (df["period"] == "Recent")]["wall_thickness"],
    df[(df["site"] == "Marsh")     & (df["period"] == "Old")]   ["wall_thickness"],
    df[(df["site"] == "Reference") & (df["period"] == "Recent")]["wall_thickness"],
    df[(df["site"] == "Reference") & (df["period"] == "Old")]   ["wall_thickness"],
]
labels = ["Marsh\nRecent", "Marsh\nOld", "Reference\nRecent", "Reference\nOld"]
colors = [MARSH_COLOR, MARSH_COLOR, REF_COLOR, REF_COLOR]

bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True,
                medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel("Wall thickness (px)")
ax.set_title("Cell Wall Thickness by Site and Period")
plt.tight_layout()
plt.savefig("data/figures/fig6_wall_thickness.png", dpi=150)
plt.close()
print("  Saved → data/figures/fig6_wall_thickness.png")

# ── Figure 7: Eccentricity by cell type ───────────────────────────────────────
print("Making Figure 7: Eccentricity by cell type...")

tracheids = df[df["cell_type"] == "tracheid"]["eccentricity"]
rays      = df[df["cell_type"] == "ray"]["eccentricity"]

fig, ax = plt.subplots(figsize=(6, 5))
ax.boxplot([tracheids, rays], tick_labels=["Tracheid", "Ray"], patch_artist=True,
           boxprops=dict(facecolor="lightcoral"),
           medianprops=dict(color="black", linewidth=2))
ax.set_ylabel("Eccentricity")
ax.set_title("Cell Eccentricity by Type\n(1 = elongated, 0 = circular)")
plt.tight_layout()
plt.savefig("data/figures/fig7_eccentricity.png", dpi=150)
plt.close()
print("  Saved → data/figures/fig7_eccentricity.png")

print("\nAll figures saved to data/figures/!")