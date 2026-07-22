# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 15:20:28 2025

@author: baujulia
"""
from Class_Single_Cell_Burst_Analyzer_force_energy import Single_Cell_Analyzer
from Class_Batch_Analyzer_force_energy import BurstBatchAnalyzer
import matplotlib.pyplot as plt
from config import X_AXIS_CONFIG


#2: Batch analysis of all files in Raw_Data folder
# =============================================================================
print("\n" + "="*70)
print("BATCH ANALYSIS")
print("="*70)

cal_path = "C:/Users/baujulia/Desktop/Desktop Folders/6 WT12 normal force/Femtotool/April 2026/Analysis/Deformation_correction wet slides/Result_Data/"

# Initialize batch analyzer
batch_analyzer = BurstBatchAnalyzer(
    base_path="C:/Users/baujulia/Desktop/Desktop Folders/6 WT12 normal force/Femtotool/April 2026/Analysis/WT12/",
    subfolder = "260417 optimal chlamy - 19 32",
    subfolder_cell_diameter = "260417 optimal chlamy - cell size - 19 32",
    file_pattern="*.txt",  # Will find all .txt files in Raw_Data folder
    save=True,
    plot=False,
    verbose=True,
    stiffness_filepath = cal_path + "System_Stiffness.txt",
    stiffness_meas_syst = None
)

# Load all data
batch_analyzer.load_all_data() 
batch_analyzer.frequency_data_aq_all()

#%% Plot Raw data
# Create individual plots for each file
# # print("\n--- Creating individual plots ---")

# batch_analyzer.plot_all_force_curves(
#     overlay = False,
#     x_mode = "displacement",
#     data=None,
#     datatype="Raw Data"
# )

# batch_analyzer.plot_all_force_curves(
#     overlay = False,
#     x_mode = "time",
#     data=None,
#     datatype="Raw Data"
# )

# ##Overlays:
    
# batch_analyzer.plot_all_force_curves(
#     overlay = True,
#     x_mode = "displacement",
#     data=None,
#     datatype="Raw Data"
# )

# batch_analyzer.plot_all_force_curves(
#     overlay = True,
#     x_mode = "time",
#     data=None,
#     datatype="Raw Data"
# )

#%% Preprocessing with Force Time

batch_analyzer.smooth_force_time_all(med_kernel = 31, sg_window= 51, sg_order= 2,plot_true = False) # Good
batch_analyzer.rolling_slope_all(window = 20,plot_true = False) # Good

#%% Peak detection and selection, baseline subtraction, normalization (start at 0 for all) with Force Time

batch_analyzer.detect_contact_slope_all(start_noise = 50, r_noise = 0.5, min_noise = 70,threshold_sigma=5,plot_true = False) # Good
batch_analyzer.subtract_baseline_all(plot_true = False) # Good

#batch_analyzer.check_normality_slopes_baseline_all() # Good, Normality confirmed most times
batch_analyzer.cutoff_all2(min_fraction=0.5,drop_limit = 20,factor_stddev = 3, plot_true=False) # True peak start as soon as backwards from peak we get inside 3 times stddev of noise, this means from thereon we are within 99.7% of noise datapoints (normal distribution), conservative. Good
batch_analyzer.shift_data_all() # start time, displacement at zero


#%% Plot all peak data

# batch_analyzer.plot_all_force_curves(
#     overlay = False,
#     x_mode = "time",
#     data="peak",
#     datatype="Peak Data"
# )

# batch_analyzer.plot_all_force_curves(
#     overlay = False,
#     x_mode = "displacement",
#     data="peak",
#     datatype="Peak Data"
# )

#%% Add single analyzers analyzed separately 

import pickle

for n in [19]:
    with open(f"saved_single_analyzers/WT12_{n}_single_analyzer.pkl", "rb") as f:
        batch_analyzer.analyzers[f"wt_{n}.txt"] = pickle.load(f)


# #%% Parameter screening for the Gaussian Sigma -> look at resulting forces and energies and save them individually

# import numpy as np
# for sigma in np.linspace(0.01, 0.1, num=10):
#     batch_analyzer.gaussian_force_displacement_all(
#                 data=None,
#                 sigma=sigma,
#                 min_points=3,
#                 compute_spread=True,
#                 plot_true=False
#                 )
    
#     batch_analyzer.calc_bursting_force_energy_all(label_add = f"Sigma_{sigma:.2f}",data_str = "Gauss", plot_true = False)
#     batch_analyzer.average_bursting_forces_displacements_energy(gauss_sigma = sigma)
    
    
# Read in the individual values of bursting force and energy from the csv files and plot them over Gaussian Sigma as Boxplots.

#%% Extraction of maximal force and energy needed to break cell. Collect the displacement at the burst.
# Choice to use Gaussian because the datapoints are not thought to be independent. 
# Choice of sigma 0.04 based on plots


batch_analyzer.gaussian_force_displacement_all(
            data=None,
            sigma=0.04,
            min_points=3,
            compute_spread=True,
            plot_true=False
            )
batch_analyzer.calc_bursting_force_energy_all(label_add = "Sigma_0.04",data_str = "Gauss", plot_true = False)
batch_analyzer.average_bursting_forces_displacements_energy(gauss_sigma = 0.04)
batch_analyzer.average_cell_diameter()

#%% Plot the Gaussian smoothed data

batch_analyzer.plot_all_force_curves(
    overlay = True,
    x_mode = "time",
    data="gauss",
    datatype="Gauss peak data",
    x_low = 0,
    x_high = 25,
    y_low = 0,
    y_high = 110
)

batch_analyzer.plot_all_force_curves(
    overlay = True,
    x_mode = "displacement",
    data="gauss",
    datatype="Gauss peak data",
    x_low = -0.2,
    x_high = 7,
    y_low = -2,
    y_high = 110
)

batch_analyzer.plot_force_mean_std(datatype="gauss", xmode="displacement", n_grid=500, min_analyzers = 10,x_low = -0.2, x_high = 4.3, y_low = -2, y_high = 46)
batch_analyzer.plot_force_mean_std(datatype="gauss", xmode="time", n_grid=500,min_analyzers =10,x_low = -0.025, x_high = 12, y_low = -0.2, y_high = 46)


#%% Save the batch analyzer for later import

import pickle
from pathlib import Path

outpath = Path("saved_batches") / "WT12_batch_analyzer.pkl"
outpath.parent.mkdir(parents=True, exist_ok=True)

with open(outpath, "wb") as f:
    pickle.dump(batch_analyzer, f)
