# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:31:07 2026

@author: baujulia
"""

from Class_Single_Cell_Burst_Analyzer_force_energy import Single_Cell_Analyzer
from Class_Batch_Analyzer_force_energy import BurstBatchAnalyzer
import matplotlib.pyplot as plt
from config import X_AXIS_CONFIG

#2: Batch analysis of all files in Raw_Data folder
# =============================================================================
print("/n" + "="*70)
print("BATCH ANALYSIS")
print("="*70)

base_path="C:/Users/baujulia/Desktop/Desktop Folders/6 WT12 normal force/Femtotool/April 2026/Analysis/Deformation_correction wet slides/"

# Initialize batch analyzer
deformation_corrector = BurstBatchAnalyzer(
    base_path=base_path,
    subfolder = "Femto",
    subfolder_cell_diameter = "Empty_cell_size",
    file_pattern="*.txt",  # Will find all .txt files in Raw_Data folder
    save=True,
    plot=False,
    verbose=True,
    stiffness_meas_syst = None,
    stiffness_filepath = None 
)

# Load all data files
deformation_corrector.load_all_data()
deformation_corrector.frequency_data_aq_all()

#%% Plot Raw data
#Create individual plots for each file
# print("\n--- Creating individual plots ---")

# deformation_corrector.plot_all_force_curves(
#     overlay = False,
#     x_mode = "displacement",
#     data=None,
#     datatype="Raw Data"
# )

# deformation_corrector.plot_all_force_curves(
#     overlay = False,
#     x_mode = "time",
#     data=None,
#     datatype="Raw Data"
# )

##Overlays:
    
deformation_corrector.plot_all_force_curves(
    overlay = True,
    x_mode = "displacement",
    data=None,
    datatype="Raw Data"
)

deformation_corrector.plot_all_force_curves(
    overlay = True,
    x_mode = "time",
    data=None,
    datatype="Raw Data"
)

#%% Preprocessing with Force Time

deformation_corrector.smooth_force_time_all(med_kernel = 21, sg_window= 31, sg_order= 2,plot_true = False) # need to detect where peak starts
deformation_corrector.rolling_slope_all(window = 20,plot_true = False) # need to detect where peak starts

#%% Peak detection and selection, baseline subtraction, normalization (start at 0 for all) with Force Time

deformation_corrector.detect_contact_slope_all(start_noise = 20, r_noise = 0.5, min_noise = 100,threshold_sigma=5,plot_true = False) # detect where the peak starts
deformation_corrector.subtract_baseline_all(plot_true = False) # subtract the baseline

deformation_corrector.cutoff_all2(min_fraction=0.5,drop_limit = 20,factor_stddev = 5, plot_true=False)
deformation_corrector.shift_data_all() # start time, displacement at zero

deformation_corrector.bin_force_displacement_all(data=None, bin_width=0.05, 
                      method="mean", min_points=1, 
                      compute_spread=False, plot_true=False)

deformation_corrector.gaussian_force_displacement_all(
            data=None,
            sigma=0.04,
            min_points=3,
            compute_spread=True,
            plot_true=False
            )

# Due to the steep increase on the glass slide, Gaussian distorts curves and binning is preferred

#%% Plot all raw peak data

deformation_corrector.plot_all_force_curves(
    overlay = False,
    x_mode = "time",
    data="peak",
    datatype="Peak Data"
)

deformation_corrector.plot_all_force_curves(
    overlay = False,
    x_mode = "displacement",
    data="peak",
    datatype="Peak Data"
)

deformation_corrector.plot_all_force_curves(
    overlay = True,
    x_mode = "time",
    data="peak",
    datatype="Peak Data"
)

deformation_corrector.plot_all_force_curves(
    overlay = True,
    x_mode = "displacement",
    data="peak",
    datatype="Peak Data"
)




#%% Based on averages make the linear fit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# ============================================================
# Build averaged force-displacement curve across experiments
# ============================================================

all_data = []

for analyzer in deformation_corrector.analyzers.values():

    tmp = analyzer.pp_binned[
        ["Displacement [um]", "Force A [uN]"]
    ].copy()
    double_tmp = tmp["Displacement [um]"]*2
    tmp["Displacement [um]"] = (double_tmp.round(2))/2 # Round displacements for almost similar ones to fall together

    all_data.append(tmp)

all_data = pd.concat(all_data, ignore_index=True)

master_curve = (
    all_data
    .groupby("Displacement [um]", as_index=False)
    .agg(
        Force_Mean=("Force A [uN]", "mean"),
        Force_STD=("Force A [uN]", "std"),
        N=("Force A [uN]", "count")
    )
)

master_curve = master_curve[    # Avoid adding points from only 2 measurements
    master_curve["N"] >= 3
]

# local_slope = np.diff(force) / np.diff(disp) # Look at local slopes, should be similar in linear range

# plt.plot(
#     disp[:-1],
#     local_slope,
#     'o-'
# )

# ============================================================
# Sort from largest displacement downward
# ============================================================

master_curve = master_curve.sort_values(
    "Displacement [um]",
    ascending=False
).reset_index(drop=True)

disp = master_curve["Displacement [um]"].values
force = master_curve["Force_Mean"].values
N = master_curve["N"].values

# ============================================================
# Progressive fitting using slope stability
# ============================================================

min_bins = 6                 # minimum bins before evaluating
slope_window = 5             # moving history length
slope_tolerance = 0.01       # 1 % slope change threshold

slope_history = []

best_idx = None
best_fit = None
best_r2 = None

for end_idx in range(min_bins, len(disp) + 1):

    x = disp[:end_idx]
    y = force[:end_idx]
    weights = N[:end_idx] # We weight by number of replicates

    slope, intercept = np.polyfit(x, y, 1, w=weights)

    y_fit = slope * x + intercept
    r2 = r2_score(y, y_fit)

    slope_history.append(slope)

    # Need enough history before checking stability
    if len(slope_history) > slope_window:

        recent_mean = np.mean(
            slope_history[-(slope_window+1):-1]
        )

        rel_change = abs(
            slope - recent_mean
        ) / abs(recent_mean)

        if rel_change > slope_tolerance:
            break

    best_idx = end_idx
    best_fit = (slope, intercept)
    best_r2 = r2

# ============================================================
# Final selected region
# ============================================================

x_linear = disp[:best_idx]
y_linear = force[:best_idx]

slope, intercept = best_fit

print(f"Slope      : {slope:.4f} uN/um")
print(f"Intercept  : {intercept:.4f} uN")
print(f"R²         : {best_r2:.5f}")
print(f"Bins used  : {best_idx}")
print(f"Datafiles  : {len(deformation_corrector.analyzers)}")

nr_datafiles = len(deformation_corrector.analyzers)

# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(7,5))

# averaged curve
plt.errorbar(
    master_curve["Displacement [um]"],
    master_curve["Force_Mean"],
    yerr=master_curve["Force_STD"],
    fmt='o',
    alpha=0.7,
    label="Mean +/- SD"
)

# selected linear region
plt.scatter(
    x_linear,
    y_linear,
    s=50,
    marker = '^',
    label="Selected linear region"
)

# fit line
x_fit = np.linspace(
    x_linear.min(),
    x_linear.max(),
    200
)

y_fit = slope * x_fit + intercept

plt.plot(
    x_fit,
    y_fit,
    linewidth=2,
    label=f"Slope = {slope:.3f} $\mu$N/$\mu$m"
)

plt.xlabel("Displacement [$\mu$m]")
plt.ylabel("Force [$\mu$N]")
plt.title("Sensor stiffness determination")

plt.text(
    0.05,
    0.6,
    f"Slope = {slope:.3f} $\mu$N/um\n"
    f"R$^2$ = {best_r2:.4f}\n"
    f"Bins = {best_idx}\n"
    f"Datafiles: {nr_datafiles}",
    transform=plt.gca().transAxes,
    va="top",
    bbox=dict(boxstyle="round", alpha=0.2)
)

plt.legend()
plt.tight_layout()

outpath = base_path + "Result_Plots/linear_regression_stiffness_wet.svg"
plt.savefig(outpath, dpi=300)
print(f"Plot saved to {outpath}")

plt.show()

#%% Export

from pathlib import Path

stiffness = slope  # [uN/um]

def export_system_stiffness(
    filepath,
    stiffness,
    intercept=None,
    r2=None,
    n_bins=None,
    n_data =None
):
    """
    Save measurement system stiffness calibration.
    """

    with open(filepath, "w") as f:
        f.write("# Measurement system calibration\n")
        f.write(f"stiffness_uN_per_um\t{stiffness:.8f}\n")

        if intercept is not None:
            f.write(f"intercept_uN\t{intercept:.8f}\n")

        if r2 is not None:
            f.write(f"r2\t{r2:.8f}\n")

        if n_bins is not None:
            f.write(f"n_bins\t{n_bins:d}\n")
            
        if n_data is not None:
            f.write(f"n_data\t{n_data:d}\n")

    print(f"Stiffness saved to: {filepath}")

filepath = base_path + "Result_Data/System_Stiffness.txt"
export_system_stiffness(
    filepath=filepath,
    stiffness=slope,
    intercept=intercept,
    r2=r2,
    n_bins=best_idx,
    n_data = nr_datafiles
)

