# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 09:00:32 2026

@author: baujulia

Mechanical properties
To apply mechanical models such as the Hertz model or capsule model to estimate the Young’s modulus of the cell (wall), indentation should stay below 10% of the total cell diameter (Yap et al., 2016). Theoretically, there is first a cell wall dominated nonlinear regime, that is for example modeled with the Hertz model or with capsule contact model. This is followed by a  turgor-dominated linear regime, where the force increases linearly with displacement. These regimes would be visible in the slope of the force curves: the Hertz model goes with indentation to the power of 3/2, so slope of 2/3*sqrt(indentation), the capsule contact model with d^3, and the linear one would be constant slope.To identify possible models that could be fitted, a linear regression approach was chosen on log(Force) vs log(relative indentation). To determine the right regression approach (mixed linear model or only population-based, first each cell was fitted individually, then it was decided if a cell-based variance was needed on the exponent as well as intercept.)

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

outdir = Path("./Mechanical_properties")
plt.rc('font', size=20)
plt.style.use(['science', 'no-latex'])
size_fig = (5,5)
colors = plt.cm.viridis(np.linspace(0, 0.9, len(batch_analyzer_WT12)))

# Regression on log(force) vs log(rel_indent)
#%% All indentation regime single plots

def plt_indentation_regimes(algae,datatype,perc,folder,low_xlim,high_xlim,low_ylim,high_ylim):
    
    if algae == "WT":
        batch = batch_analyzer_WT12
    elif algae == "chl":
        batch = batch_analyzer_chlorella
    else:
        print("Enter valid algae: WT or chl")
        return
        
    
    for name, analyzer in batch.analyzers.items():
        if datatype == "gauss":
            data = analyzer.pp_gauss
            
        elif datatype == "binned":
            data = analyzer.pp_binned
            
        elif datatype == "peak":
            data = analyzer.peak_data
            
        else:
            print("Enter valid datatype: gauss, binned, or peak")
            break
        
        
        displacement = data["Displacement [um]"].values
        rel_indent = displacement/(analyzer.cell_diameter/2)
        force = data["Force A [uN]"].values        
        
        
        force_ln = np.log(force)
        rel_indent_ln = np.log(rel_indent)

        
        rel_indent_below_10_perc_ln = rel_indent_ln[rel_indent<= perc]
        force_below_10_perc_ln = force_ln[rel_indent<=perc]
        
        
        plt.figure(figsize = size_fig)
        plt.plot(
            rel_indent[rel_indent<=perc],
            force[rel_indent<=perc],
            color=colors[0],
            linestyle = "",
            marker = "o"
            )
        plt.xlabel("Rel. indentation")
        plt.ylabel("Force ($\mu$N)")
        plt.tight_layout()
        
        
        plt.xscale('log')
        plt.yscale('log')
        
        
        plt.ylim(low_ylim,high_ylim)
        plt.xlim(low_xlim,high_xlim)
        
        outpath = outdir / folder / f"{name}_force {datatype}_rel_indent_{perc}_ln_scale.svg"
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

#%% Execution

plt_indentation_regimes("WT","gauss",0.1,"Small_indent",0.01,0.1,0.05,2)
plt_indentation_regimes("chl","gauss",0.1, "Small_indent",0.01,0.1,0.01,5)

plt_indentation_regimes("WT","gauss",1,"Large_indent",0.1,1,0.05,100)
plt_indentation_regimes("chl","gauss",1,"Large_indent",0.1,1,0.05,100)

#%% Stiffness all



def plt_stiffness(algae,datatype,perc,folder,low_xlim,high_xlim,low_ylim,high_ylim,sg_window_length,sg_order):
    from scipy.signal import savgol_filter
    if algae == "WT":
        batch = batch_analyzer_WT12
    elif algae == "chl":
        batch = batch_analyzer_chlorella
    else:
        print("Enter valid algae: WT or chl")
        return
        
    
    for name, analyzer in batch.analyzers.items():
        if datatype == "gauss":
            data = analyzer.pp_gauss
            
        elif datatype == "binned":
            data = analyzer.pp_binned
            
        elif datatype == "peak":
            data = analyzer.peak_data
            
        else:
            print("Enter valid datatype: gauss, binned, or peak")
            break
        
        
        displacement = data["Displacement [um]"].values
        rel_indent = displacement/(analyzer.cell_diameter/2)
        force = data["Force A [uN]"].values        
        
        local_stiffness_savgol = savgol_filter(force, window_length=sg_window_length,
                  polyorder=sg_order, deriv=1, delta=np.mean(np.diff(displacement)))
        
        
        outpath = outdir / folder / f"{name}_local_stiffness {datatype}_rel_indent_{perc}.svg"
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        
        plt.figure(figsize = size_fig)
        plt.plot(
            rel_indent[rel_indent<=perc],
            local_stiffness_savgol[rel_indent<=perc],
            color=colors[0],
            linestyle = "",
            marker = "o"
            )
        plt.xlabel("Rel. indentation")
        plt.ylabel("Stiffness ($\mu$N/$\mu$m)")
        plt.tight_layout()
        
        
        plt.xscale('log')
        
        
        plt.ylim(low_ylim,high_ylim)
        plt.xlim(low_xlim,high_xlim)
        
        outpath = outdir / folder / f"{name}_savgol_stiffness {datatype}_rel_indent_{perc}.svg"
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        
plt_stiffness("WT","gauss",1,"Stiffness",0.01,1,-1,30,31,2)
plt_stiffness("chl","gauss",1,"Stiffness",0.01,1,-1,30,31,2)

plt_stiffness("WT","gauss",0.1,"Stiffness",0.01,0.1,-2,4,31,2)
plt_stiffness("chl","gauss",0.1,"Stiffness",0.001,0.1,-2,4,31,2)

#%% Overlays WT settings

algae = "wt"
wt6 = batch_analyzer_WT12.analyzers["wt_6.txt"]
wt11 = batch_analyzer_WT12.analyzers["wt_11.txt"]
wt23 = batch_analyzer_WT12.analyzers["wt_23.txt"]

selected_analyzers = [wt6,wt11,wt23]
colors = plt.cm.viridis(np.linspace(0, 0.9, 3))

#%% Plot WT6, WT23 and WT11 in Small indentation overlay figure

perc = 0.1

# Savgol settings
from scipy.signal import savgol_filter
sg_window_length = 11
sg_order = 2

plt.figure(figsize = size_fig)
for i, analyzer in enumerate(selected_analyzers):
    
    data = analyzer.pp_gauss
    displacement = data["Displacement [um]"].values
    rel_indent = displacement/(analyzer.cell_diameter/2)
    force = data["Force A [uN]"].values        
    local_stiffness = np.diff(data["Force A [uN]"].values)/np.diff(data["Displacement [um]"].values)
    local_stiffness_savgol = savgol_filter(force, window_length=sg_window_length,
              polyorder=sg_order, deriv=1, delta=np.mean(np.diff(displacement)))

    plt.plot(
        rel_indent[rel_indent<=perc]/2,
        force[rel_indent<=perc],
        color=colors[i],
        linestyle = "",
        marker = "o",
        label = f"replicate {i+1}"
        )
    
plt.xlabel("Rel. displacement")
plt.ylabel("Force ($\mu$N)")
plt.legend()
plt.tight_layout()

plt.xscale('log')
plt.yscale('log')
ax = plt.gca()
plt.setp(ax.get_xticklabels(which='both'),
         rotation=45,
         ha='right')

plt.ylim(0.1,2)
plt.xlim(0.01,0.05)

outpath = outdir / "Publication_Plots" / f"{algae}_overlay_log_log_gauss_{perc}.svg"
outpath.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(outpath, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

#%% Plot WT6,11,23   in Large indentation overlay figure
perc = 1

plt.figure(figsize = size_fig)
for i, analyzer in enumerate(selected_analyzers):
    
    data = analyzer.pp_gauss
    displacement = data["Displacement [um]"].values
    rel_indent = displacement/(analyzer.cell_diameter/2)
    force = data["Force A [uN]"].values        

    plt.plot(
        rel_indent[rel_indent<=perc]/2,
        force[rel_indent<=perc],
        color=colors[i],
        linestyle = "",
        marker = "o",
        label = f"replicate {i+1}"
        )
    
plt.xlabel("Rel. displacement")
plt.ylabel("Force ($\mu$N)")
plt.legend()
plt.tight_layout()

plt.xscale('log')
plt.yscale('log')
ax = plt.gca()
plt.setp(ax.get_xticklabels(which='both'),
         rotation=45,
         ha='right')

plt.ylim(1,100)
plt.xlim(0.1,1)
outpath = outdir / "Publication_Plots" / f"{algae}_overlay_log_log_gauss_{perc}.svg"
outpath.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(outpath, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

#%% Plot the stiffness (local) over relative displacement for WT6,11,23  in Large indentation overlay figure

from scipy.signal import savgol_filter

perc = 1
sg_window_length = 101
sg_order = 2

plt.figure(figsize = size_fig)
for i, analyzer in enumerate(selected_analyzers):
    
    data = analyzer.pp_gauss
    displacement = data["Displacement [um]"].values
    rel_indent = displacement/(analyzer.cell_diameter/2)
    force = data["Force A [uN]"].values        
    local_stiffness_savgol = savgol_filter(force, window_length=sg_window_length,
              polyorder=sg_order, deriv=1, delta=np.mean(np.diff(displacement)))

    plt.plot(
        rel_indent[rel_indent<=perc]/2,
        local_stiffness_savgol[rel_indent<=perc],
        color=colors[i],
        linestyle = "",
        marker = "o",
        label = f"replicate {i+1}"
        )
    
plt.xlabel("Rel. displacement")
plt.ylabel("Stiffness ($\mu$N/$\mu$m)")
plt.legend()
plt.tight_layout()

plt.xscale('log')

ax = plt.gca()
plt.setp(ax.get_xticklabels(which='both'),
         rotation=45,
         ha='right')

plt.ylim(0,60)
plt.xlim(0.1,1)
outpath = outdir / "Publication_Plots" / f"{algae}_overlay_stiffness_{perc}.svg"
outpath.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(outpath, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

#%% Plot the stiffness (local) over relative displacement for WT6, WT23 and WT11 in Small indentation overlay figure

perc = 0.1
sg_window_length = 51
sg_order = 2

plt.figure(figsize = size_fig)
for i, analyzer in enumerate(selected_analyzers):
    
    data = analyzer.pp_gauss
    displacement = data["Displacement [um]"].values
    rel_indent = displacement/(analyzer.cell_diameter/2)
    force = data["Force A [uN]"].values        
    local_stiffness_savgol = savgol_filter(force, window_length=sg_window_length,
              polyorder=sg_order, deriv=1, delta=np.mean(np.diff(displacement)))

    plt.plot(
        rel_indent[rel_indent<=perc]/2,
        local_stiffness_savgol[rel_indent<=perc],
        color=colors[i],
        linestyle = "",
        marker = "o",
        label = f"replicate {i+1}"
        )
    
plt.xlabel("Rel. displacement")
plt.ylabel("Stiffness ($\mu$N/$\mu$m)")
plt.legend()
plt.tight_layout()

plt.xscale('log')

ax = plt.gca()
plt.setp(ax.get_xticklabels(which='both'),
         rotation=45,
         ha='right')

plt.ylim(0,6)
plt.xlim(0.005,0.05)
outpath = outdir / "Publication_Plots" / f"{algae}_overlay_stiffness_{perc}.svg"
outpath.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(outpath, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

#%% Overlays Chl settings

algae = "chl"
chl6 = batch_analyzer_chlorella.analyzers["chl_6.txt"]
chl_26 = batch_analyzer_chlorella.analyzers["chl_26.txt"]
chl_60 = batch_analyzer_chlorella.analyzers["chl_60.txt"]

selected_analyzers = [chl6,chl_26,chl_60]
colors = plt.cm.viridis(np.linspace(0, 0.9, 3))

#%% Plot exemplary chl in Small indentation overlay figure

perc = 0.1

# Savgol settings
from scipy.signal import savgol_filter
sg_window_length = 11
sg_order = 2

plt.figure(figsize = size_fig)
for i, analyzer in enumerate(selected_analyzers):
    
    data = analyzer.pp_gauss
    displacement = data["Displacement [um]"].values
    rel_indent = displacement/(analyzer.cell_diameter/2)
    force = data["Force A [uN]"].values        
    local_stiffness = np.diff(data["Force A [uN]"].values)/np.diff(data["Displacement [um]"].values)
    local_stiffness_savgol = savgol_filter(force, window_length=sg_window_length,
              polyorder=sg_order, deriv=1, delta=np.mean(np.diff(displacement)))

    plt.plot(
        rel_indent[rel_indent<=perc]/2,
        force[rel_indent<=perc],
        color=colors[i],
        linestyle = "",
        marker = "o",
        label = f"replicate {i+1}"
        )
    
plt.xlabel("Rel. displacement")
plt.ylabel("Force ($\mu$N)")
plt.legend()
plt.tight_layout()

plt.xscale('log')
plt.yscale('log')
ax = plt.gca()
plt.setp(ax.get_xticklabels(which='both'),
         rotation=45,
         ha='right')

plt.ylim(0.05,1)
plt.xlim(0.01,0.05)

outpath = outdir / "Publication_Plots" / f"{algae}_overlay_log_log_gauss_{perc}.svg"
outpath.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(outpath, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

#%% Plt chlorella exemplary overlay
perc = 1

plt.figure(figsize = size_fig)
for i, analyzer in enumerate(selected_analyzers):
    
    data = analyzer.pp_gauss
    displacement = data["Displacement [um]"].values
    rel_indent = displacement/(analyzer.cell_diameter/2)
    force = data["Force A [uN]"].values        

    plt.plot(
        rel_indent[rel_indent<=perc]/2,
        force[rel_indent<=perc],
        color=colors[i],
        linestyle = "",
        marker = "o",
        label = f"replicate {i+1}"
        )
    
plt.xlabel("Rel. displacement")
plt.ylabel("Force ($\mu$N)")
plt.legend()
plt.tight_layout()

plt.xscale('log')
plt.yscale('log')
ax = plt.gca()
plt.setp(ax.get_xticklabels(which='both'),
         rotation=45,
         ha='right')

plt.ylim(0.4,30)
plt.xlim(0.1,1)
outpath = outdir / "Publication_Plots" / f"{algae}_overlay_log_log_gauss_{perc}.svg"
outpath.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(outpath, dpi=300, bbox_inches="tight")
plt.show()
plt.close()


#%% Median ± IQR overlay plots with theoretical model lines.

"""
Median ± IQR overlay plots with theoretical model lines.

Model lines (not fitted, shape only — anchored to data median at a reference point):
  Small indentation (<10% radius):
    - Hertz contact:           F ∝ δ^(3/2)
    - Capsule contact model:   F ∝ δ^3
  Large indentation (10–100% radius):
    - Turgor-dominated linear: F ∝ δ^1
"""
 
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
 
 
# ── helpers ────────────────────────────────────────────────────────────────────
 
def build_rel_indent_force_matrix(batch, datatype, perc, n_grid=300, min_cells=10):
    """
    Interpolate each cell's (rel_indent, force) onto a common log-spaced grid
    up to `perc` of cell radius. Returns (x_common, matrix) where matrix rows
    are individual cells.
 
    rel_indent is defined as displacement / (cell_diameter / 2),
    i.e. normalised to cell radius — the standard for Hertz / capsule models.
    """
    x_min_global, x_max_global = np.inf, -np.inf
 
    # --- first pass: find global rel_indent range ---
    for analyzer in batch.analyzers.values():
        data = _get_data(analyzer, datatype)
        displacement = data["Displacement [um]"].values
        rel_indent = displacement / (analyzer.cell_diameter / 2)
        force = data["Force A [uN]"].values
 
        valid = np.isfinite(rel_indent) & np.isfinite(force) & (rel_indent > 0) & (force > 0)
        ri = rel_indent[valid & (rel_indent <= perc)]
        if len(ri) < 2:
            continue
        x_min_global = min(x_min_global, ri.min())
        x_max_global = max(x_max_global, ri.max())
 
    # log-spaced grid (natural for log-log plots)
    x_common = np.logspace(np.log10(x_min_global), np.log10(x_max_global), n_grid)
 
    # --- second pass: interpolate ---
    rows = []
    for analyzer in batch.analyzers.values():
        data = _get_data(analyzer, datatype)
        displacement = data["Displacement [um]"].values
        rel_indent = displacement / (analyzer.cell_diameter / 2)
        force = data["Force A [uN]"].values
 
        valid = (np.isfinite(rel_indent) & np.isfinite(force)
                 & (rel_indent >= 0) & (force > 0) & (rel_indent < perc))
        ri = rel_indent[valid]
        f  = force[valid]
 
        if len(ri) < 2:
            continue
 
        sort_idx = np.argsort(ri)
        ri = ri[sort_idx]
        f  = f[sort_idx]
 
        interp_fn = interp1d(ri, f, kind="linear",
                             bounds_error=False, fill_value=np.nan)
        rows.append(interp_fn(x_common))
 
    matrix = np.array(rows)   # shape (n_cells, n_grid)
 
    # mask positions with too few valid contributions
    valid_counts = np.sum(np.isfinite(matrix), axis=0)
    enough = valid_counts >= min_cells
    matrix[:, ~enough] = np.nan
 
    return x_common[enough], matrix[:, enough]
 
 
def compute_median_iqr(matrix):
    """Returns (median, q25, q75) arrays, NaN-aware."""
    median = np.nanmedian(matrix, axis=0)
    q25    = np.nanpercentile(matrix, 25, axis=0)
    q75    = np.nanpercentile(matrix, 75, axis=0)
    return median, q25, q75
 
 
def _get_data(analyzer, datatype):
    if datatype == "gauss":
        return analyzer.pp_gauss
    elif datatype == "binned":
        return analyzer.pp_binned
    elif datatype == "peak":
        return analyzer.peak_data
    else:
        raise ValueError("datatype must be: gauss, binned, or peak")
 
 
def model_line(x, exponent, x_ref, y_ref):
    """
    Power-law model line anchored to pass through (x_ref, y_ref).
    F = y_ref * (x / x_ref)^exponent
    This shows the *shape* of the model without implying a fitted prefactor.
    """
    return y_ref * (x / x_ref) ** exponent
 
 
# ── main plotting function ──────────────────────────────────────────────────────
 
def plt_median_iqr_log_log(
    algae,
    datatype,
    perc,
    folder,
    low_xlim, high_xlim,
    low_ylim,  high_ylim,
    n_grid=300,
    min_cells=10,
    show_models=True,
    size_fig=(5, 5.5),
):
    """
    Plots the population median force with IQR band (25–75 %) in log-log space,
    plus theoretical model lines anchored to the median at the geometric centre
    of the x-axis range.
 
    Parameters
    ----------
    algae       : "WT" or "chl"
    datatype    : "gauss", "binned", or "peak"
    perc        : upper rel_indent cutoff (e.g. 0.1 for small, 1 for large)
    folder      : sub-folder name inside outdir
    low/high_*  : axis limits (in data units, not log)
    show_models : whether to overlay model lines
    """
 
    if algae == "WT":
        batch = batch_analyzer_WT12
        color = plt.cm.viridis(0.25)
        label_str = "WT12"
    elif algae == "chl":
        batch = batch_analyzer_chlorella
        color = plt.cm.viridis(0.75)
        label_str = "Chlorella"
    else:
        print("Enter valid algae: WT or chl")
        return
 
    x_common, matrix = build_rel_indent_force_matrix(
        batch, datatype, perc, n_grid=n_grid, min_cells=min_cells
    )
 
    if x_common.size == 0:
        print(f"No valid data for {algae}, {datatype}, perc={perc}")
        return
 
    median, q25, q75 = compute_median_iqr(matrix)
 
    # --- anchor point for model lines: geometric centre of x range ---
    x_ref = np.sqrt(low_xlim * high_xlim)
    # find closest grid index
    anchor_idx = np.argmin(np.abs(x_common - x_ref))
    y_ref = median[anchor_idx]
 
    fig, ax = plt.subplots(figsize=size_fig)
 
    # IQR band
    ax.fill_between(x_common, q25, q75,
                    color=color, alpha=0.3, label="IQR (25-75 %)")
    # Median
    ax.plot(x_common, median,
            color=color, linewidth=2, label=f"Median ({label_str})")
 
    # --- model lines ---
    if show_models:
        x_model = np.logspace(np.log10(low_xlim), np.log10(high_xlim), 200)
 
        if perc <= 0.15:
            # small indentation regime
            # Hertz contact: F ∝ δ^(3/2)  →  exponent = 1.5
            y_hertz = model_line(x_model, 1.5, x_ref, y_ref)
            ax.plot(x_model, y_hertz, "--", color="black",
                    linewidth=1.5, label=r"Hertz: $F \propto \delta^{3/2}$")
 
            # Thin shell model: F ∝ δ^0.5
            y_capsule = model_line(x_model, 0.5, x_ref, y_ref)
            ax.plot(x_model, y_capsule, ":", color="black",
                    linewidth=1.5, label=r"Thin-shell: $F \propto \delta^{0.5}$")
 
        else:
            # large indentation — turgor-dominated linear regime: F ∝ δ^1
            y_linear = model_line(x_model, 1.0, x_ref, y_ref)
            ax.plot(x_model, y_linear, "--", color="black",
                    linewidth=1.5, label=r"Turgor (linear): $F \propto \delta$")
            
            # large indentation — isobaric regime: F ∝ δ^4/3
            y_linear = model_line(x_model, 1.333333, x_ref, y_ref)
            ax.plot(x_model, y_linear, ":", color="black",
                    linewidth=1.5, label=r"Isobaric: $F \propto \delta^{4/3}$")
 
    n_cells = matrix.shape[0]
    ax.set_xlabel("Rel. indentation ($\delta$ / R)")
    ax.set_ylabel("Force ($\mu$N)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low_xlim, high_xlim)
    ax.set_ylim(low_ylim, high_ylim)
    ax.legend(fontsize=12, frameon=True)
    ax.set_title(f"n = {n_cells} cells", fontsize=12)
 
    plt.setp(ax.get_xticklabels(which="both"), rotation=45, ha="right")
    plt.tight_layout()
 
    outpath = outdir / folder / f"{algae}_median_iqr_log_log_{datatype}_perc{perc}.svg"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    
    
def plt_median_iqr_log_log_overlay(
    datatype,
    perc,
    folder,
    low_xlim, high_xlim,
    low_ylim,  high_ylim,
    n_grid=300,
    min_cells=10,
    show_models=True,
    size_fig=(5, 5.5),
):
    """
    Plots the population median force with IQR band (25–75 %) in log-log space,
    plus theoretical model lines anchored to the median at the geometric centre
    of the x-axis range for both algae in overlay.
 
    Parameters
    ----------
    algae       : "WT" or "chl"
    datatype    : "gauss", "binned", or "peak"
    perc        : upper rel_indent cutoff (e.g. 0.1 for small, 1 for large)
    folder      : sub-folder name inside outdir
    low/high_*  : axis limits (in data units, not log)
    show_models : whether to overlay model lines
    """
 
    x_common_wt, matrix_wt = build_rel_indent_force_matrix(
        batch_analyzer_WT12, datatype, perc, n_grid=n_grid, min_cells=min_cells
    )
    
    x_common_chl, matrix_chl = build_rel_indent_force_matrix(
        batch_analyzer_chlorella, datatype, perc, n_grid=n_grid, min_cells=min_cells
    )
 
    median_wt, q25_wt, q75_wt = compute_median_iqr(matrix_wt)
    median_chl, q25_chl, q75_chl = compute_median_iqr(matrix_chl)
 
    # --- anchor point for model lines: geometric centre of x range of Wild Type ---
    x_ref = np.sqrt(low_xlim * high_xlim)
    # find closest grid index
    anchor_idx = np.argmin(np.abs(x_common_wt - x_ref))
    y_ref = (median_wt[anchor_idx] + median_chl[anchor_idx])/2
 
    fig, ax = plt.subplots(figsize=size_fig)
 
    # IQR band
    ax.fill_between(x_common_wt, q25_wt, q75_wt,
                    color=plt.cm.viridis(0.25), alpha=0.3, label="IQR $C.$ $reinhardtii$ (25-75 %)")
    # Median
    ax.plot(x_common_wt, median_wt,
            color=plt.cm.viridis(0.25), linewidth=2, label=f"Median $C.$ $reinhardtii$")
    
    # IQR band
    ax.fill_between(x_common_chl, q25_chl, q75_chl,
                    color=plt.cm.viridis(0.75), alpha=0.3, label="IQR $C.$ $sorokiniana$ (25-75 %)")
    # Median
    ax.plot(x_common_chl, median_chl,
            color=plt.cm.viridis(0.75), linewidth=2, label=f"Median $C.$ $sorokiniana$")
 
    # --- model lines ---
    if show_models:
        x_model = np.logspace(np.log10(low_xlim), np.log10(high_xlim), 200)
 
        if perc <= 0.15:
            # small indentation regime
            # Hertz contact: F ∝ δ^(3/2)  →  exponent = 1.5
            y_hertz = model_line(x_model, 1.5, x_ref, y_ref)
            ax.plot(x_model, y_hertz, "--", color="black",
                    linewidth=1.5, label=r"Hertz: $F \propto \delta^{3/2}$")
 
            # Thin shell model: F ∝ δ^0.5
            y_capsule = model_line(x_model, 0.5, x_ref, y_ref)
            ax.plot(x_model, y_capsule, ":", color="black",
                    linewidth=1.5, label=r"Thin-shell: $F \propto \delta^{0.5}$")
 
        else:
            # large indentation — turgor-dominated linear regime: F ∝ δ^1
            y_linear = model_line(x_model, 1.0, x_ref, y_ref)
            ax.plot(x_model, y_linear, "--", color="black",
                    linewidth=1.5, label=r"Linear: $F \propto \delta$")
            
            # large indentation — isobaric regime: F ∝ δ^4/3
            y_linear = model_line(x_model, 1.333333, x_ref, y_ref)
            ax.plot(x_model, y_linear, ":", color="black",
                    linewidth=1.5, label=r"Isobaric: $F \propto \delta^{4/3}$")
 
    ax.set_xlabel("Rel. indentation ($\delta$ / R)")
    ax.set_ylabel("Force ($\mu$N)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low_xlim, high_xlim)
    ax.set_ylim(low_ylim, high_ylim)
    ax.legend(fontsize=16, frameon=True)
 
    plt.setp(ax.get_xticklabels(which="both"), rotation=45, ha="right")
    plt.tight_layout()
 
    outpath = outdir / folder / f"overlay_median_iqr_log_log_{datatype}_perc{perc}.svg"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
 
 
# ── execution ──────────────────────────────────────────────────────────────────
 
# Small indentation
plt_median_iqr_log_log("WT",  "gauss", 0.1, "Median_IQR", 0.01, 0.1,  0.05, 2)
plt_median_iqr_log_log("chl", "gauss", 0.1, "Median_IQR", 0.01, 0.1,  0.01, 5)
 
# Large indentation
plt_median_iqr_log_log("WT",  "gauss", 1.0, "Median_IQR", 0.1,  1.0,  0.05, 100)
plt_median_iqr_log_log("chl", "gauss", 1.0, "Median_IQR", 0.1,  1.0,  0.05, 100)

# Overlay large indentation
plt_median_iqr_log_log_overlay(
    "gauss",
    1.0,
    "Median_IQR",
    0.1, 1,
    0.05,  100,
    n_grid=300,
    min_cells=10,
    show_models=True,
    size_fig=(5, 5.5),
)

# Overlay small indentation
plt_median_iqr_log_log_overlay(
    "gauss",
    0.1,
    "Median_IQR",
    0.01, 0.1,
    0.05,  100,
    n_grid=300,
    min_cells=10,
    show_models=True,
    size_fig=(5, 5.5),
)

#%% Mechanical properties 
"""
Mechanical properties — exponent fitting and local slope analysis
=================================================================
Per-cell log(F) vs log(δ/R) power-law fitting, population statistics,
Mann-Whitney U comparison between WT12 (Chlamydomonas) and Chlorella,
and local slope (d log F / d log δ) aggregated across cells.

Key changes vs previous version:
  - fit_exponent_per_cell now takes perc_low AND perc_high, so large
    indentation is fitted over [0.1, 1.0] rather than [0, 1.0], keeping
    the two regimes genuinely independent.
  - Added data_density_report() to inspect how many points and what
    log(δ/R) span each cell contributes per window — critical for judging
    whether small-indentation fits are trustworthy.
  - min_log_span added as a validity criterion alongside min_points.

Depends on: batch_analyzer_WT12, batch_analyzer_chlorella, outdir, size_fig
being defined in the parent script (Mechanical_properties.py).
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu, linregress
from scipy.signal import savgol_filter


# ══════════════════════════════════════════════════════════════════════════════
# 0.  Data density diagnostics
# ══════════════════════════════════════════════════════════════════════════════

def data_density_report(batch, datatype, perc_low, perc_high, label=""):
    """
    For each cell, reports how many data points fall in [perc_low, perc_high]
    and what log10 span of δ/R is covered.

    This is a sanity check before fitting: a regression on 10 points spanning
    half a decade of x is much less reliable than one on 200 points spanning
    two decades, even if both pass a min_points threshold.

    Prints a summary and returns a DataFrame for inspection.
    """
    records = []
    for name, analyzer in batch.analyzers.items():
        data      = _get_data(analyzer, datatype)
        disp      = data["Displacement [um]"].values
        ri        = disp / (analyzer.cell_diameter / 2)
        force     = data["Force A [uN]"].values

        mask = (ri > perc_low) & (ri <= perc_high) & (force > 0) \
               & np.isfinite(ri) & np.isfinite(force)
        ri_w = ri[mask]

        if len(ri_w) < 2:
            log_span = np.nan
        else:
            log_span = np.log10(ri_w.max()) - np.log10(ri_w.min())

        records.append({
            "name":      name,
            "n_points":  mask.sum(),
            "log_span":  log_span,        # decades of δ/R covered
            "ri_min":    ri_w.min() if len(ri_w) else np.nan,
            "ri_max":    ri_w.max() if len(ri_w) else np.nan,
        })

    df = pd.DataFrame(records)
    window_str = f"δ/R ∈ ({perc_low}, {perc_high}]"
    print(f"\n── Data density: {label}  {window_str} ──")
    print(f"  n cells with any data : {(df['n_points'] > 0).sum()}")
    print(f"  n_points  median [IQR]: "
          f"{df['n_points'].median():.0f}  "
          f"[{df['n_points'].quantile(0.25):.0f}, "
          f"{df['n_points'].quantile(0.75):.0f}]")
    print(f"  n_points  min / max   : "
          f"{df['n_points'].min()} / {df['n_points'].max()}")
    print(f"  log₁₀ span  median    : {df['log_span'].median():.2f} decades")
    print(f"  log₁₀ span  min / max : "
          f"{df['log_span'].min():.2f} / {df['log_span'].max():.2f}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Per-cell power-law fitting
# ══════════════════════════════════════════════════════════════════════════════

def fit_exponent_per_cell(batch, datatype,
                          perc_low, perc_high,
                          min_points=10,
                          min_log_span=0.5):
    """
    Fits log(F) = n * log(δ/R) + const for each cell individually
    within the indentation window (perc_low, perc_high].

    Use perc_low=0,   perc_high=0.1  for the small-indentation regime.
    Use perc_low=0.1, perc_high=1.0  for the large-indentation regime.

    Validity criteria (cells not meeting these are excluded and reported):
      min_points   : minimum number of data points in the window
      min_log_span : minimum log10 span of δ/R covered (decades).
                     A fit over less than 0.5 decades is unreliable regardless
                     of point count — the x-range is too narrow to constrain
                     the slope.

    Returns a DataFrame with one row per included cell:
        name, exponent, intercept, r_squared, std_err, n_points, log_span

    Excluded cells are printed as a warning so you know what was dropped
    and why — they should not be silently discarded.
    """
    records  = []
    excluded = []

    for name, analyzer in batch.analyzers.items():
        data  = _get_data(analyzer, datatype)
        disp  = data["Displacement [um]"].values
        ri    = disp / (analyzer.cell_diameter / 2)
        force = data["Force A [uN]"].values

        mask = (ri > perc_low) & (ri <= perc_high) & (force > 0) \
               & np.isfinite(ri) & np.isfinite(force)
        ri_w = ri[mask]
        f_w  = force[mask]

        n      = len(ri_w)
        span   = (np.log10(ri_w.max()) - np.log10(ri_w.min())) \
                 if n >= 2 else 0.0

        if n < min_points:
            excluded.append((name, f"only {n} points (need {min_points})"))
            continue
        if span < min_log_span:
            excluded.append((name,
                f"log span {span:.2f} dec < {min_log_span} dec"))
            continue

        sort_idx = np.argsort(ri_w)
        ri_w = ri_w[sort_idx]
        f_w  = f_w[sort_idx]

        log_ri = np.log(ri_w)
        log_f  = np.log(f_w)

        slope, intercept, r, _, stderr = linregress(log_ri, log_f)

        records.append({
            "name":      name,
            "exponent":  slope,
            "intercept": intercept,
            "r_squared": r ** 2,
            "std_err":   stderr,
            "n_points":  n,
            "log_span":  span,
        })

    if excluded:
        print(f"\n  ⚠ Excluded cells (window {perc_low}–{perc_high}):")
        for nm, reason in excluded:
            print(f"    {nm}: {reason}")

    return pd.DataFrame(records)


def check_normality_exponent_distr(dataframe):
    import statsmodels.api as sm
    data = dataframe
    exponents = data["exponent"]
    fig = sm.qqplot(exponents, line = '45')
    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Population statistics + Mann-Whitney
# ══════════════════════════════════════════════════════════════════════════════

def compare_exponents(df_wt, df_chl, perc_low, perc_high, folder, outdir):
    """
    Summary statistics and Mann-Whitney U test on per-cell exponents,
    plus a strip + box plot saved to disk.
    """

    small_indent_params = [(1.5, "Hertz: $F \propto \delta^{3/2}$", "--"),
                         (0.5, "Thin-shell: $F \propto \delta^{0.5}$",":")]
    
    large_indent_params = [(1, "Linear: $F \propto \delta$", "--"),
                         (1.333333, "Isobaric: $F \propto \delta^{4/3}$",":")]

    def summary(df, label):
        n = df["exponent"].dropna()
        print(f"\n── {label} (n={len(n)} cells) ──")
        print(f"  Median exponent : {n.median():.3f}")
        print(f"  IQR             : [{n.quantile(0.25):.3f}, "
              f"{n.quantile(0.75):.3f}]")
        print(f"  Mean ± SD       : {n.mean():.3f} ± {n.std():.3f}")
        print(f"  Range           : [{n.min():.3f}, {n.max():.3f}]")
        print(f"  Median R²       : {df['r_squared'].median():.3f}")
        print(f"  Median n_points : {df['n_points'].median():.0f}")
        print(f"  Median log_span : {df['log_span'].median():.2f} decades")

    summary(df_wt,  "WT12 (Chlamydomonas)")
    summary(df_chl, "Chlorella")

    stat, p = mannwhitneyu(
        df_wt["exponent"].dropna(),
        df_chl["exponent"].dropna(),
        alternative="two-sided"
    )
    print(f"\nMann-Whitney U = {stat:.1f},  p = {p:.4f}")
    p_label = f"p = {p:.3f}" if p >= 0.001 else "p $<$ 0.001"

    # ── strip + box plot ──
    fig, ax = plt.subplots(figsize=(5, 6))

    groups   = [df_wt["exponent"].dropna().values,
                df_chl["exponent"].dropna().values]
    labels   = ["$C.$ $reinhardtii$", "$C.$ $sorokiniana$"]
    colors_g = [plt.cm.viridis(0.25), plt.cm.viridis(0.75)]

    for i, (vals, col) in enumerate(zip(groups, colors_g)):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=col, alpha=0.6, s=30, zorder=3)
        ax.boxplot(vals, positions=[i], widths=0.35,
                   patch_artist=True, zorder=2,
                   medianprops=dict(color= col, linewidth=2),
                   boxprops=dict(facecolor=col, edgecolor = "none", alpha=0.3),
                   whiskerprops=dict(linewidth=1.2, color = col),
                   capprops=dict(linewidth=1.2, color = col),
                   flierprops=dict(marker="", color = col))
    if perc_high<0.15:
        params = small_indent_params
    else:
        params = large_indent_params
    for exp, lbl, ls in params:
        ax.axhline(exp, linestyle=ls, color="black", linewidth=1.2, label=lbl)

    y_max     = max(groups[0].max(), groups[1].max())
    y_bracket = y_max * 1.05
    ax.plot([0, 0, 1, 1],
            [y_bracket, y_bracket + 0.03, y_bracket + 0.03, y_bracket],
            color="black", linewidth=1.2)
    ax.text(0.5, y_bracket + 0.04, p_label,
            ha="center", va="bottom", fontsize=16)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels,rotation = 45)
    ax.set_ylabel("Power-law exponent")
    ax.set_title(f"$\delta$/R in ({perc_low}, {perc_high}]")
    ax.legend(fontsize=16, frameon=True)
    plt.tight_layout()

    tag = f"{perc_low}_{perc_high}"
    outpath = outdir / folder / f"exponent_comparison_{tag}.svg"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    return stat, p


# ══════════════════════════════════════════════════════════════════════════════
# 3.  R² and n_points distributions
# ══════════════════════════════════════════════════════════════════════════════

def plot_fit_quality(df_wt, df_chl, perc_low, perc_high, folder, outdir,
                     r2_threshold=0.7):
    """
    Two-panel figure:
      Left  — histogram of R² per cell type
      Right — histogram of n_points per cell type

    Showing n_points alongside R² makes the trustworthiness of each
    fit transparent: a high R² on very few points is not reliable.

    r2_threshold is shown as a reference line; 0.7 is a reasonable
    minimum for a phenomenological fit reported in a publication.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 5))

    bins_r2 = np.linspace(0, 1, 21)
    bins_n  = np.linspace(0,
                          max(df_wt["n_points"].max(),
                              df_chl["n_points"].max()) + 10,
                          20)

    for df, label, col in [(df_wt,  "$C.$ $reinhardtii$",      plt.cm.viridis(0.25)),
                            (df_chl, "$C.$ $sorokiniana$", plt.cm.viridis(0.75))]:
        axes[0].hist(df["r_squared"], bins=bins_r2,
                     alpha=0.6, color=col, label=label)
        axes[1].hist(df["n_points"],  bins=bins_n,
                     alpha=0.6, color=col, label=label)

    axes[0].axvline(r2_threshold, color="black", linestyle="--",
                    linewidth=1.2, label=f"R$^2$ = {r2_threshold}")
    axes[0].set_xlabel("R$^2$  (log-log fit)")
    axes[0].set_ylabel("Number of cells")
    axes[0].legend(fontsize=16)

    axes[1].set_xlabel("Number of points for fit")
    axes[1].set_ylabel("Number of cells")
    axes[1].legend(fontsize=16)

    tag = f"{perc_low}_{perc_high}"
    fig.suptitle(f"Fit quality - $\delta$/R in ({perc_low}, {perc_high}]",
                 fontsize=16)
    plt.tight_layout()

    outpath = outdir / folder / f"fit_quality_{tag}.svg"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    for label, df in [("$C.$ $reinhardtii$", df_wt), ("$C.$ $sorokiniana$", df_chl)]:
        frac = (df["r_squared"] >= r2_threshold).mean()
        print(f"{label}: {frac*100:.0f} % of cells have R² ≥ {r2_threshold}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Local slope  d(log F)/d(log δ)  aggregated across cells
# ══════════════════════════════════════════════════════════════════════════════

def build_local_slope_matrix(batch, datatype, perc_low, perc_high,
                             sg_window_length=21, sg_order=2,
                             n_grid=200, min_cells=5):
    """
    Computes local log-log slope d(logF)/d(log δ) per cell using a
    Savitzky-Golay derivative on log-uniformly-resampled data,
    then interpolates onto a common log-spaced grid.

    perc_low / perc_high bound the δ/R window, consistent with fitting.
    """
    from scipy.interpolate import interp1d

    x_min_g, x_max_g = np.inf, -np.inf
    for analyzer in batch.analyzers.values():
        data  = _get_data(analyzer, datatype)
        disp  = data["Displacement [um]"].values
        ri    = disp / (analyzer.cell_diameter / 2)
        force = data["Force A [uN]"].values
        mask  = (ri > perc_low) & (ri <= perc_high) & (force > 0) \
                & np.isfinite(ri) & np.isfinite(force)
        ri_w  = ri[mask]
        if len(ri_w) < sg_window_length + 1:
            continue
        x_min_g = min(x_min_g, ri_w.min())
        x_max_g = max(x_max_g, ri_w.max())

    if np.isinf(x_min_g):
        return np.array([]), np.array([])

    x_common = np.logspace(np.log10(x_min_g), np.log10(x_max_g), n_grid)
    rows = []

    for analyzer in batch.analyzers.values():
        data  = _get_data(analyzer, datatype)
        disp  = data["Displacement [um]"].values
        ri    = disp / (analyzer.cell_diameter / 2)
        force = data["Force A [uN]"].values

        mask = (ri > perc_low) & (ri <= perc_high) & (force > 0) \
               & np.isfinite(ri) & np.isfinite(force)
        ri_w = ri[mask]
        f_w  = force[mask]

        if len(ri_w) < sg_window_length + 1:
            continue

        sort_idx = np.argsort(ri_w)
        ri_w = ri_w[sort_idx]
        f_w  = f_w[sort_idx]

        log_ri = np.log(ri_w)
        log_f  = np.log(f_w)

        # resample onto uniform log spacing before SG (SG assumes uniform spacing)
        log_ri_uniform = np.linspace(log_ri.min(), log_ri.max(), len(log_ri))
        interp_logf    = interp1d(log_ri, log_f, kind="linear",
                                  bounds_error=False, fill_value=np.nan)# Builds model on log_ri and log_f
        log_f_uniform  = interp_logf(log_ri_uniform) # returns model values at log_ri_uniform
        delta_log_ri   = np.mean(np.diff(log_ri_uniform))

        local_slope = savgol_filter(
            log_f_uniform,
            window_length=sg_window_length,
            polyorder=sg_order,
            deriv=1,
            delta=delta_log_ri
        )
        # interpolate slope back onto common grid
        interp_slope = interp1d(
            np.exp(log_ri_uniform), local_slope,
            kind="linear", bounds_error=False, fill_value=np.nan
        )  # Builds model on ri vs local slope
        rows.append(interp_slope(x_common)) # use model to predict for x_common ri values

    if not rows:
        return x_common, np.full((0, n_grid), np.nan)

    matrix = np.array(rows)
    valid_counts = np.sum(np.isfinite(matrix), axis=0)
    matrix[:, valid_counts < min_cells] = np.nan

    return x_common, matrix


def plt_local_slope_comparison(datatype, perc_low, perc_high, folder,
                                low_xlim, high_xlim,
                                low_ylim=0.0, high_ylim=2.5,
                                sg_window_length=21, sg_order=2,
                                n_grid=200, min_cells=10,
                                size_fig=(4.5, 5.5)):
    """
    Median ± IQR of the local log-log slope for both species on one axes.
    """
    fig, ax = plt.subplots(figsize=size_fig)

    configs = [
        (batch_analyzer_WT12,      "WT12 (Chlamydomonas)", plt.cm.viridis(0.25)),
        (batch_analyzer_chlorella, "Chlorella",            plt.cm.viridis(0.75)),
    ]

    for batch, label, color in configs:
        x_common, matrix = build_local_slope_matrix(
            batch, datatype, perc_low, perc_high,
            sg_window_length=sg_window_length,
            sg_order=sg_order,
            n_grid=n_grid,
            min_cells=min_cells
        )
        if x_common.size == 0:
            continue

        median = np.nanmedian(matrix, axis=0)
        q25    = np.nanpercentile(matrix, 25, axis=0)
        q75    = np.nanpercentile(matrix, 75, axis=0)

        ax.fill_between(x_common, q25, q75, color=color, alpha=0.25)
        ax.plot(x_common, median, color=color, linewidth=2, label=label)

    for exp, lbl, ls in [(1.0, "linear / turgor (1)", "--"),
                         (1.5, "Hertz (3/2)",          ":")]:
        ax.axhline(exp, linestyle=ls, color="grey", linewidth=1.2, label=lbl)

    ax.set_xscale("log")
    ax.set_xlim(low_xlim, high_xlim)
    ax.set_ylim(low_ylim, high_ylim)
    ax.set_xlabel("Rel. indentation ($\delta$ / R)")
    ax.set_ylabel(r"Local slope  $d\,\log F\,/\,d\,\log\delta$")
    ax.set_title(f"Local log-log slope - $\delta$/R in ({perc_low}, {perc_high}]")
    ax.legend(fontsize=11, frameon=True)
    plt.setp(ax.get_xticklabels(which="both"), rotation=45, ha="right")
    plt.tight_layout()

    tag = f"{perc_low}_{perc_high}"
    outpath = outdir / folder / f"local_slope_comparison_{datatype}_{tag}.svg"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════════

def _get_data(analyzer, datatype):
    if datatype == "gauss":
        return analyzer.pp_gauss
    elif datatype == "binned":
        return analyzer.pp_binned
    elif datatype == "peak":
        return analyzer.peak_data
    else:
        raise ValueError("datatype must be: gauss, binned, or peak")


# ══════════════════════════════════════════════════════════════════════════════
# execution
# ══════════════════════════════════════════════════════════════════════════════

DATATYPE = "gauss"
FOLDER   = "Exponent_Analysis"

# ── 0. Data density check FIRST — read before interpreting any fit results ───

print("\n════════ Data density: small indentation (0, 0.1] ════════")
dd_wt_s  = data_density_report(batch_analyzer_WT12,
                                DATATYPE, 0, 0.1, label="WT12")
dd_chl_s = data_density_report(batch_analyzer_chlorella,
                                DATATYPE, 0, 0.1, label="Chlorella")

print("\n════════ Data density: large indentation (0.1, 1.0] ════════")
dd_wt_l  = data_density_report(batch_analyzer_WT12,
                                DATATYPE, 0.1, 1.0, label="WT12")
dd_chl_l = data_density_report(batch_analyzer_chlorella,
                                DATATYPE, 0.1, 1.0, label="Chlorella")

# ── 1. Per-cell exponent fits ─────────────────────────────────────────────────

# Small indentation: (0, 0.1]
# min_log_span=0.3 is already less than 1 decade — inspect density report
# to decide if even this is achievable for all cells.
print("\n════════ Exponent fits: small indentation (0, 0.1] ════════")
df_wt_small  = fit_exponent_per_cell(batch_analyzer_WT12,
                                      DATATYPE, 0, 0.1,
                                      min_points=10, min_log_span=0.3)
df_chl_small = fit_exponent_per_cell(batch_analyzer_chlorella,
                                      DATATYPE, 0, 0.1,
                                      min_points=10, min_log_span=0.3)



stat_s, p_s = compare_exponents(df_wt_small, df_chl_small,
                                 0, 0.1, folder=FOLDER, outdir=outdir)
plot_fit_quality(df_wt_small, df_chl_small,
                 0, 0.1, folder=FOLDER, outdir=outdir)

# Large indentation: (0.1, 1.0]  — genuinely independent of small regime
print("\n════════ Exponent fits: large indentation (0.1, 1.0] ════════")
df_wt_large  = fit_exponent_per_cell(batch_analyzer_WT12,
                                      DATATYPE, 0.1, 1.0,
                                      min_points=10, min_log_span=0.5)
df_chl_large = fit_exponent_per_cell(batch_analyzer_chlorella,
                                      DATATYPE, 0.1, 1.0,
                                      min_points=10, min_log_span=0.5)

stat_l, p_l = compare_exponents(df_wt_large, df_chl_large,
                                 0.1, 1.0, folder=FOLDER, outdir=outdir)
plot_fit_quality(df_wt_large, df_chl_large,
                 0.1, 1.0, folder=FOLDER, outdir=outdir)


# Check if exponents are normally distributed
check_normality_exponent_distr(df_wt_small)
check_normality_exponent_distr(df_chl_small)
check_normality_exponent_distr(df_wt_large)
check_normality_exponent_distr(df_chl_large)

# ── 2. Local slope plots ──────────────────────────────────────────────────────

plt_local_slope_comparison(DATATYPE, 0, 0.1, FOLDER,
                            low_xlim=0.01, high_xlim=0.1,
                            low_ylim=0.0,  high_ylim=2.5,
                            sg_window_length=11, sg_order=2)

# Smoothing with Savgol only from 0.1 to 1
plt_local_slope_comparison(DATATYPE, 0.1, 1.0, FOLDER,
                            low_xlim=0.1,  high_xlim=1.0,
                            low_ylim= 0.0,  high_ylim=4,
                            sg_window_length=51, sg_order=2)

# Smoothing with Savgol on whole indentation range
plt_local_slope_comparison(DATATYPE, 0, 1.0, FOLDER,
                            low_xlim=0.1,  high_xlim=1.0,
                            low_ylim= 0.0,  high_ylim=4,
                            sg_window_length=51, sg_order=2)

# ── 3. Save exponent tables ───────────────────────────────────────────────────

for label, df in [("WT12_small",     df_wt_small),
                  ("Chlorella_small", df_chl_small),
                  ("WT12_large",      df_wt_large),
                  ("Chlorella_large", df_chl_large)]:
    p = outdir / FOLDER / f"exponents_{label}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"Saved: {p}")