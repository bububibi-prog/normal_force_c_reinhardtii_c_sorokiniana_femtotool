# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 15:31:29 2026

@author: baujulia
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import scienceplots

base_path = "C:/Users/baujulia/Desktop/Desktop Folders/6 WT12 normal force/Femtotool/April 2026/Analysis/"

with open(base_path + "saved_batches/WT12_batch_analyzer.pkl", "rb") as f:
    batch_analyzer_WT12 = pickle.load(f)
with open(base_path + "saved_batches/Chlorella_batch_analyzer.pkl", "rb") as f:
    batch_analyzer_chlorella = pickle.load(f)

outdir = Path("./Batch_Comparison_Plots")


plt.rc('font', size=20)
plt.style.use(['science', 'no-latex'])

#%% Plot Overlay force vs displacement
from config import X_AXIS_CONFIG
cmap = plt.cm.viridis
colors = cmap([0.25, 0.75])


def _compute_median_iqr(batch_analyzer, datatype="gauss", xmode="displacement",
                         n_grid=500, min_analyzers=21):
    """
    Helper: interpolates all curves in a single batch_analyzer onto a common
    x-grid and returns the median/IQR statistics (only where enough curves
    contribute, per min_analyzers).
 
    Returns
    -------
    x_common, median_force, iqr_lower, iqr_upper : np.ndarray
    """
    from scipy.interpolate import interp1d
 
    if xmode not in X_AXIS_CONFIG:
        raise ValueError(f"xmode must be one of {list(X_AXIS_CONFIG.keys())}")
 
    forces_interp = []
    x_min, x_max = np.inf, -np.inf
 
    def _get_data(analyzer):
        if datatype == "gauss":
            return analyzer.pp_gauss
        elif datatype == "binned":
            return analyzer.pp_binned
        elif datatype == "peak":
            return analyzer.peak_data
        else:
            raise ValueError("datatype must be: gauss, binned, peak")
 
    # -------------------------------------------------
    # 1. First pass: determine global x-range
    # -------------------------------------------------
    for analyzer in batch_analyzer.analyzers.values():
        data = _get_data(analyzer)
 
        if xmode == "displacement":
            x = data["Displacement [um]"]
        else:
            x = data["Time [s]"]
 
        x_min = min(x_min, np.nanmin(x))
        x_max = max(x_max, np.nanmax(x))
 
    # common grid
    x_common = np.linspace(x_min, x_max, n_grid)
 
    # -------------------------------------------------
    # 2. Second pass: interpolate each curve
    # -------------------------------------------------
    for name, analyzer in batch_analyzer.analyzers.items():
        data = _get_data(analyzer)
 
        # ----- select x data -----
        if xmode == "displacement":
            x = data["Displacement [um]"]
        else:
            x = data["Time [s]"]
 
        y = data["Force A [uN]"]
 
        # safety: remove NaNs
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
 
        # sort (required for interpolation)
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]
 
        # skip bad curves
        if len(x) < 2:
            continue
 
        # interpolation function
        f = interp1d(
            x,
            y,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan
        )
 
        forces_interp.append(f(x_common))
 
    forces_interp = np.array(forces_interp)
 
    # -------------------------------------------------
    # 3. Compute statistics (ignore NaNs)
    # -------------------------------------------------
 
    # number of valid (non-NaN) contributions at each x
    valid_counts = np.sum(np.isfinite(forces_interp), axis=0)
 
    # enforce minimum support
    valid_mask = valid_counts >= min_analyzers
 
    median_force = np.full(forces_interp.shape[1], np.nan)
    median_force[valid_mask] = np.nanmedian(forces_interp[:, valid_mask], axis=0)
    median_force = median_force[valid_mask]
 
    q25 = np.nanpercentile(forces_interp[:, valid_mask], 25, axis=0)
    q75 = np.nanpercentile(forces_interp[:, valid_mask], 75, axis=0)
 
    iqr_lower = q25
    iqr_upper = q75
 
    x_common = x_common[valid_mask]
 
    return x_common, median_force, iqr_lower, iqr_upper

def plot_force_median_iqr_overlay(datatype="gauss", xmode="displacement", n_grid=500,
                                 min_analyzers=21, x_low=-0.025, x_high=4.25,
                                 y_low=-0.2, y_high=46):
    """
    Plots median +/- IQR force curves for WT12 and Chlorella batch analyzers
    overlaid on the same axes. WT12 uses colors[0] (viridis 0.25),
    Chlorella uses colors[1] (viridis 0.75).
    """
 
    datasets = [
        ("WT12", batch_analyzer_WT12, colors[0]),
        ("Chlorella", batch_analyzer_chlorella, colors[1]),
    ]
 
    plt.figure(figsize=(5, 5))
 
    for label, batch_analyzer, color in datasets:
        x_common, median_force, iqr_lower, iqr_upper = _compute_median_iqr(
            batch_analyzer,
            datatype=datatype,
            xmode=xmode,
            n_grid=n_grid,
            min_analyzers=min_analyzers,
        )
 
        plt.plot(
            x_common,
            median_force,
            color=color,
            label=f"{label} Median"
        )
 
        plt.fill_between(
            x_common,
            iqr_lower,
            iqr_upper,
            color=color,
            alpha=0.3,
            label=f"{label} IQR (25-75%)"
        )
 
    plt.xlabel(X_AXIS_CONFIG[xmode]["label"])
    plt.ylabel("Force ($\\mu$N)")
    plt.legend()
    plt.xlim(x_low, x_high)
    plt.ylim(y_low, y_high)
    plt.tight_layout()
 
    outpath = outdir / f"median_IQR_overlay_{datatype}_{xmode}.svg"
    plt.savefig(outpath, dpi=300)
    plt.show()
    plt.close()

plot_force_median_iqr_overlay(datatype="gauss", xmode="displacement",n_grid=500,min_analyzers = 10,x_low = -0.025, x_high = 4.3, y_low = -0.2, y_high = 46)

plot_force_median_iqr_overlay(datatype="gauss", xmode="time",n_grid=500,min_analyzers = 10,x_low = -0.025, x_high = 12, y_low = -0.2, y_high = 46)

#%% Boxplots bursting parameters
def extract_param(batch_analyzer, attr_name):
    """
    Extract a parameter from all Single_Cell_Analyzer objects in a batch.
    """
    values = []
    for analyzer in batch_analyzer.analyzers.values():
        val = getattr(analyzer, attr_name, None)
        if val is not None and not np.isnan(val):
            values.append(val)
    return np.array(values)


def boxplot_two_batches(
    batch1,
    batch2,
    label1="C. reinhardtii WT12",
    label2="C. sorokiniana",
    param="bursting_force",
    ylabel="",
    title="",
    outpath=None,
    figsize=(5, 5),
):
    """
    Create a two-group boxplot with jittered datapoints,
    using viridis colors for dots and median lines.
    """

    data1 = extract_param(batch1, param)
    data2 = extract_param(batch2, param)

    fig, ax = plt.subplots(figsize=figsize)

    # Viridis colors (avoid extreme ends)
    cmap = plt.cm.viridis
    colors = cmap([0.25, 0.75])

    # --- Jittered datapoints ---
    jitter = 0.05

    ax.scatter(
        np.ones(len(data1)) + jitter * (np.random.rand(len(data1)) - 0.5),
        data1,
        color=colors[0],
        alpha=0.5,
        s=30,
        zorder=3,
    )

    ax.scatter(
        2 * np.ones(len(data2)) + jitter * (np.random.rand(len(data2)) - 0.5),
        data2,
        color=colors[1],
        alpha=0.5,
        s=30,
        zorder=3,
    )

    # --- Boxplots ---
    bp = ax.boxplot(
        [data1, data2],
        positions=[1, 2],
        widths=0.5,
        patch_artist=True,
        showfliers=False,
    )

    # Color boxes and medians
    for i in range(2):
        bp["boxes"][i].set_facecolor(colors[i])
        bp["boxes"][i].set_alpha(0.2)
        bp["boxes"][i].set_edgecolor(colors[i])

        bp["medians"][i].set_color(colors[i])
        bp["medians"][i].set_linewidth(2)

        bp["whiskers"][2*i].set_color(colors[i])
        bp["whiskers"][2*i + 1].set_color(colors[i])
        bp["caps"][2*i].set_color(colors[i])
        bp["caps"][2*i + 1].set_color(colors[i])

    # --- Axes formatting ---
    ax.set_xticks([1, 2])
    ax.set_xticklabels([label1, label2])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_box_aspect(1)

    plt.tight_layout()

    if outpath is not None:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()
    
    


# Cell diameter
boxplot_two_batches(
    batch_analyzer_WT12,
    batch_analyzer_chlorella,
    label1="WT12",
    label2="Chlorella",
    param="cell_diameter",
    ylabel="Cell diameter ($\mu$m)",
    title="",
    outpath=outdir / "Cell_Diameter_Comparison.svg",
)

# Bursting force
boxplot_two_batches(
    batch_analyzer_WT12,
    batch_analyzer_chlorella,
    label1="WT12",
    label2="Chlorella",
    param="bursting_force",
    ylabel="Bursting force ($\mu$N)",
    title="",
    outpath=outdir / "Bursting_Force_Comparison.svg",
)

# Bursting energy
boxplot_two_batches(
    batch_analyzer_WT12,
    batch_analyzer_chlorella,
    label1="WT12",
    label2="Chlorella",
    param="bursting_energy",
    ylabel="Bursting energy (pJ)",
    title="",
    outpath=outdir / "Bursting_Energy_Comparison.svg",
)


# Absolute Bursting displacement
boxplot_two_batches(
    batch_analyzer_WT12,
    batch_analyzer_chlorella,
    label1="WT12",
    label2="Chlorella",
    param="bursting_displacement",
    ylabel="Bursting displacement ($\mu$m)",
    title="",
    outpath=outdir / "Bursting_Displacement_Comparison.svg",
)

# Relative Bursting displacement
boxplot_two_batches(
    batch_analyzer_WT12,
    batch_analyzer_chlorella,
    label1="WT12",
    label2="Chlorella",
    param="bursting_displacement_per_diameter",
    ylabel="Relative bursting displacement",
    title="",
    outpath=outdir / "Relative_Bursting_Displacement_Comparison.svg",
)

# Energy per dry weight
boxplot_two_batches(
    batch_analyzer_WT12,
    batch_analyzer_chlorella,
    label1="WT12",
    label2="Chlorella",
    param="energy_dw",
    ylabel="Energy per dry weight (kJ/kg)",
    title="",
    outpath=outdir / "Bursting_Energy_per_dw_Comparison.svg",
)




#%% Statistics 
import scipy.stats as stats
from scipy.stats import mannwhitneyu, levene, ttest_ind, shapiro
import warnings
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

outdir = Path("./Batch_Comparison_Plots")
outdir.mkdir(parents=True, exist_ok=True)

params = [
    "cell_diameter",
    "bursting_force",
    "bursting_energy",
    "bursting_displacement",
    "bursting_displacement_per_diameter",
    "energy_dw",
]

lines = []
def log(s=""):
    print(s)
    lines.append(s)

def run_tests(data_wt, data_chl, scale_label):
    """
    Run Shapiro-Wilk, Levene, Welch t-test, Cohen's d, and Mann-Whitney
    on the provided (already-transformed) arrays. Returns a dict of results.
    """
    sw_w_wt,  sw_p_wt  = shapiro(data_wt)
    sw_w_chl, sw_p_chl = shapiro(data_chl)
    both_normal = (sw_p_wt >= 0.05) and (sw_p_chl >= 0.05)

    lev_w, lev_p = levene(data_wt, data_chl)

    t_stat, t_p = ttest_ind(data_wt, data_chl, equal_var=False)
    
    # Pooled variance according to: J. Hartung, G. Knapp, B. K. Sinha: Statistical Meta-Analysis with Application. Wiley, New Jersey 2008, ISBN 978-0-470-29089-7.
    pooled_std = np.sqrt(
        ((len(data_wt)  - 1) * np.std(data_wt,  ddof=1)**2
       + (len(data_chl) - 1) * np.std(data_chl, ddof=1)**2)
        / (len(data_wt) + len(data_chl) - 2)
    )
    
    # J. Cohen: Statistical Power Analysis for the Behavioral Sciences. 2. Auflage. Lawrence Erlbaum Associates, Hillsdale 1988, ISBN 0-8058-0283-5.
    
    cohens_d = (np.mean(data_wt) - np.mean(data_chl)) / pooled_std
    magnitude = ("negligible" if abs(cohens_d) < 0.2 else
                 "small"      if abs(cohens_d) < 0.5 else
                 "medium"     if abs(cohens_d) < 0.8 else 
                 "large")

    mw_stat, mw_p = mannwhitneyu(data_wt, data_chl, alternative="two-sided")

    log(f"  [{scale_label}]")
    log(f"    Shapiro-Wilk WT12:      W={sw_w_wt:.4f},  p={sw_p_wt:.4f}"
        + ("  → normal" if sw_p_wt >= 0.05 else "  → NON-NORMAL"))
    log(f"    Shapiro-Wilk Chlorella: W={sw_w_chl:.4f},  p={sw_p_chl:.4f}"
        + ("  → normal" if sw_p_chl >= 0.05 else "  → NON-NORMAL"))
    if not both_normal:
        log(f"    ⚠ Normality violated — Welch result unreliable; rely on Mann-Whitney.")
    log(f"    Levene's test:          W={lev_w:.4f},  p={lev_p:.4f}"
        + ("  → variances differ" if lev_p < 0.05 else "  → variances homogeneous"))
    log(f"    Welch's t-test:         t={t_stat:.4f},  p={t_p:.4f}"
        + ("  → SIGNIFICANT" if t_p < 0.05 else "  → not significant"))
    log(f"    Cohen's d:              d={cohens_d:.4f}  ({magnitude})")
    log(f"    Mann-Whitney U:         U={mw_stat:.1f},  p={mw_p:.4f}"
        + ("  → SIGNIFICANT" if mw_p < 0.05 else "  → not significant"))

    if (t_p < 0.05) != (mw_p < 0.05):
        log(f"    ⚠ Welch and Mann-Whitney disagree — inspect carefully.")

    return dict(
        sw_p_wt=sw_p_wt, sw_p_chl=sw_p_chl, both_normal=both_normal,
        lev_p=lev_p, t_p=t_p, cohens_d=cohens_d, mw_p=mw_p
    )


log("=" * 72)
log("Statistical comparison: C. reinhardtii WT12 vs C. sorokiniana")
log("All tests two-sided, alpha = 0.05")
log("=" * 72)

for param in params:
    param_wt  = extract_param(batch_analyzer_WT12, param)
    param_chl = extract_param(batch_analyzer_chlorella, param)

    if len(param_wt) == 0 or len(param_chl) == 0:
        log(f"\n{param}: No data found, skipping.")
        continue

    log(f"\n{'─'*72}")
    log(f"PARAMETER: {param}")
    log(f"  WT12:      n={len(param_wt)},  "
        f"median={np.median(param_wt):.4g},  "
        f"IQR=[{np.percentile(param_wt,25):.4g}, {np.percentile(param_wt,75):.4g}]")
    log(f"  Chlorella: n={len(param_chl)},  "
        f"median={np.median(param_chl):.4g},  "
        f"IQR=[{np.percentile(param_chl,25):.4g}, {np.percentile(param_chl,75):.4g}]")
    log("")

    # --- Linear scale ---
    res_lin = run_tests(param_wt, param_chl, "linear scale")
    log("")

    # --- Log10 scale ---
    log_wt  = np.log10(param_wt)
    log_chl = np.log10(param_chl)
    res_log = run_tests(log_wt, log_chl, "log10 scale")
    log("")

    # --- Recommended test summary ---
    lin_ok  = res_lin["both_normal"]
    log_ok  = res_log["both_normal"]
    if log_ok:
        recommend = "log10-scale Welch's t-test (log-normality confirmed)"
    elif lin_ok:
        recommend = "linear-scale Welch's t-test (linear normality confirmed)"
    else:
        recommend = "Mann-Whitney U test (normality not confirmed on either scale)"
    log(f"  => Recommended test for reporting: {recommend}")

    # --- QQ plots: linear and log, per group ---
    for group_data, group_label in [(param_wt, "WT12"), (param_chl, "Chlorella")]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"{param} — {group_label}", fontsize=13)

        stats.probplot(group_data, dist="norm", plot=axes[0])
        axes[0].set_title("Linear scale")

        stats.probplot(np.log10(group_data), dist="norm", plot=axes[1])
        axes[1].set_title("Log10 scale")

        plt.tight_layout()
        outpath = outdir / f"QQ_{param}_{group_label}.svg"
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()


log("\n" + "=" * 72)

# Save results to text file
result_path = outdir / "statistical_results.txt"
with open(result_path, "w") as f:
    f.write("\n".join(lines))
print(f"\nResults saved to: {result_path}")