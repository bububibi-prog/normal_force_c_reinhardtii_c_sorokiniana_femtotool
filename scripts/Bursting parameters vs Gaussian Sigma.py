# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:39:41 2026

@author: baujulia
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METRICS = [
    "bursting_force_uN",
    "bursting_energy_pJ",
    "bursting_displacement_um",
    "energy_per_dw_J_per_kg",
    "bursting_force_per_diameter_uN_per_um",
    "bursting_energy_per_diameter_pJ_per_um",
    "bursting_displacement_per_diameter_um_per_um",
]


def _parse_sigma(filename: str) -> float | None:
    """Extract sigma value from filename like 'bursting_individual_values_Gaussian_Sigma_0.05.csv'."""
    m = re.search(r"bursting_individual_values_Gaussian_Sigma[_\s]*([\d.]+)", filename, re.IGNORECASE)
    return float(m.group(1)) if m else None


def load_sigma_csvs(folder: Path) -> dict[float, pd.DataFrame]:
    """
    Load all CSVs matching the sigma-sweep naming convention from *folder*.
    Returns a dict mapping sigma -> DataFrame.
    """
    folder = Path(folder)
    data = {}
    for csv_path in sorted(folder.glob("*.csv")):
        sigma = _parse_sigma(csv_path.name)
        if sigma is None:
            continue
        df = pd.read_csv(csv_path)
        data[sigma] = df
    if not data:
        raise FileNotFoundError(f"No sigma CSV files found in {folder}")
    return data


def boxplot_sigma_sweep(
    folder: Path,
    metric: str = "bursting_force_uN",
    ylabel: str = "",
    title: str = "",
    outpath: Path | None = None,
    figsize: tuple = (7, 5),
) -> None:
    """
    Boxplot of *metric* across Gaussian sigma values loaded from CSV files in *folder*.

    Each dot represents one sample (cell_id). Boxes show IQR, whiskers are 1.5×IQR.
    Uses the viridis colour scheme, publication-ready styling.

    Parameters
    ----------
    folder   : Path to directory containing the sigma-sweep CSV files.
    metric   : Column name to plot. One of METRICS.
    ylabel   : Y-axis label (defaults to the metric name if empty).
    title    : Figure title.
    outpath  : If given, figure is saved here (SVG/PNG/PDF inferred from suffix).
               Parent directories are created automatically.
    figsize  : (width, height) in inches.
    """
    
    plt.rc('font', size=20)
    plt.style.use(['science', 'no-latex'])
    
    if metric not in METRICS:
        raise ValueError(f"Unknown metric '{metric}'. Choose from:\n  " + "\n  ".join(METRICS))

    sigma_data = load_sigma_csvs(folder)
    sigmas = sorted(sigma_data.keys())
    n = len(sigmas)

    if n == 0:
        raise ValueError("No data loaded.")

    # --- Viridis colour per sigma position ---
    cmap = plt.cm.viridis
    norm_positions = np.linspace(0.1, 0.9, n)
    colors = [cmap(p) for p in norm_positions]

    # --- Gather per-sigma value arrays ---
    values = []
    for s in sigmas:
        col = sigma_data[s][metric].dropna().values.astype(float)
        values.append(col)

    # --- Figure ---
    fig, ax = plt.subplots(figsize=figsize)

    positions = list(range(1, n + 1))
    jitter_width = 0.08

    # Jittered scatter dots
    rng = np.random.default_rng(42)
    for i, (vals, pos, c) in enumerate(zip(values, positions, colors)):
        jitter = rng.uniform(-jitter_width, jitter_width, size=len(vals))
        ax.scatter(
            pos + jitter,
            vals,
            color=c,
            alpha=0.55,
            s=28,
            zorder=3,
            linewidths=0,
        )

    # Boxplots
    bp = ax.boxplot(
        values,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        zorder=2,
    )

    for i, c in enumerate(colors):
        bp["boxes"][i].set_facecolor(c)
        bp["boxes"][i].set_alpha(0.20)
        bp["boxes"][i].set_edgecolor(c)
        bp["medians"][i].set_color(c)
        bp["medians"][i].set_linewidth(2)
        bp["whiskers"][2 * i].set_color(c)
        bp["whiskers"][2 * i + 1].set_color(c)
        bp["caps"][2 * i].set_color(c)
        bp["caps"][2 * i + 1].set_color(c)

    # --- Axes formatting ---
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{s:.2f}" for s in sigmas], rotation=45, ha="right")
    ax.set_xlabel("Gaussian sigma")
    ax.set_ylabel(ylabel if ylabel else metric.replace("_", " "))
    if title:
        ax.set_title(title, pad=8)

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.tick_params(axis="both")

    # Colourbar legend showing sigma → viridis mapping
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(sigmas), vmax=max(sigmas)))
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
    # cbar.set_label("Gaussian sigma")
    # cbar.ax.tick_params()

    plt.tight_layout()

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        print(f"Saved → {outpath}")

    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results_folder = Path("C:/Users/baujulia/Desktop/Desktop Folders/6 WT12 normal force/Femtotool/April 2026/Analysis/WT12/Batch_Results")
    output_path    = Path("C:/Users/baujulia/Desktop/Desktop Folders/6 WT12 normal force/Femtotool/April 2026/Analysis/WT12/Batch_Plots/bursting_force_vs_sigma.svg")

    boxplot_sigma_sweep(
        folder=results_folder,
        metric="bursting_force_uN",
        ylabel="Bursting force (µN)",
        title="Bursting force vs. Gaussian smoothing sigma",
        outpath=output_path,
    )
    
if __name__ == "__main__":
    results_folder = Path("C:/Users/baujulia/Desktop/Desktop Folders/6 WT12 normal force/Femtotool/April 2026/Analysis/WT12/Batch_Results")
    output_path    = Path("C:/Users/baujulia/Desktop/Desktop Folders/6 WT12 normal force/Femtotool/April 2026/Analysis/WT12/Batch_Plots/bursting_energy_vs_sigma.svg")

    boxplot_sigma_sweep(
        folder=results_folder,
        metric="bursting_energy_pJ",
        ylabel="Bursting energy (pJ)",
        title="Bursting energy vs. Gaussian smoothing sigma",
        outpath=output_path,
    )
    
    
if __name__ == "__main__":
    results_folder = Path("C:/Users/baujulia/Desktop/Desktop Folders/6 WT12 normal force/Femtotool/April 2026/Analysis/Chlorella/Batch_Results")
    output_path    = Path("C:/Users/baujulia/Desktop/Desktop Folders/6 WT12 normal force/Femtotool/April 2026/Analysis/Chlorella/Batch_Plots/bursting_force_vs_sigma.svg")

    boxplot_sigma_sweep(
        folder=results_folder,
        metric="bursting_force_uN",
        ylabel="Bursting force (µN)",
        title="Bursting force vs. Gaussian smoothing sigma",
        outpath=output_path,
    )
    
if __name__ == "__main__":
    results_folder = Path("C:/Users/baujulia/Desktop/Desktop Folders/6 WT12 normal force/Femtotool/April 2026/Analysis/Chlorella/Batch_Results")
    output_path    = Path("C:/Users/baujulia/Desktop/Desktop Folders/6 WT12 normal force/Femtotool/April 2026/Analysis/Chlorella/Batch_Plots/bursting_energy_vs_sigma.svg")

    boxplot_sigma_sweep(
        folder=results_folder,
        metric="bursting_energy_pJ",
        ylabel="Bursting energy (pJ)",
        title="Bursting energy vs. Gaussian smoothing sigma",
        outpath=output_path,
    )


    # --- Other metrics, same folder ---
    # boxplot_sigma_sweep(results_folder, metric="bursting_energy_pJ",
    #                     ylabel="Bursting energy (pJ)",
    #                     outpath=output_path.parent / "bursting_energy_vs_sigma.svg")