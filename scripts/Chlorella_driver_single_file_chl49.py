# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 14:50:15 2026

@author: baujulia
"""

from Class_Single_Cell_Burst_Analyzer_force_energy import Single_Cell_Analyzer
import matplotlib.pyplot as plt
from config import X_AXIS_CONFIG


#%%2: Analysis of single file
# =============================================================================
print("\n" + "="*70)
print("SINGLE FILE ANALYSIS")
print("="*70)

filename = "chl_49.txt"
subfolder = "260417 optimal chlorella/"
subfolder_cell_diameter = "Cell size optimal 260417 chlorella/"
base_path = "C:/Users/baujulia/Desktop/Desktop Folders/6 WT12 normal force/Femtotool/April 2026/Analysis/Chlorella/"
cal_path = "C:/Users/baujulia/Desktop/Desktop Folders/6 WT12 normal force/Femtotool/April 2026/Analysis/Deformation_correction wet slides/Result_Data/"

# Initialize batch analyzer
analyzer = Single_Cell_Analyzer(
    filename=filename,
    subfolder = subfolder,
    subfolder_cell_diameter=subfolder_cell_diameter,
    base_path=base_path,
    save=True,
    plot=False,  # Suppress individual plots during batch load
    verbose=False,# Suppress individual verbose output
    stiffness_meas_syst = None,
    stiffness_filepath = cal_path + "System_Stiffness.txt" # To be determined with only glass slide
)


#%% Plot Raw data
# Create individual plots for each file
# # print("\n--- Creating individual plots ---")

analyzer.plot_force_curve(x_mode="displacement", 
                          data="raw", 
                          datatype=None)

analyzer.plot_force_curve(x_mode="time", 
                          data="raw", 
                          datatype=None)


#%% Preprocessing with Force Time

analyzer.smooth_force_time(med_kernel = 21, sg_window= 31, sg_order= 2,plot_true = True) # Good
analyzer.rolling_slope(window = 20,plot_true = True) # Good

#%% Peak detection and selection, baseline subtraction, normalization (start at 0 for all) with Force Time

analyzer.detect_contact_slope(start_noise = 30, r_noise = 0.25, min_noise = 70,threshold_sigma=5,plot_true = True) # Good
analyzer.subtract_baseline(plot_true = True) # Good

analyzer.check_normality_slopes_baseline() # Good, Normality confirmed
analyzer.cutoff2(min_fraction=0.1,drop_limit = 5,factor_stddev = 3, plot_true=True) # True peak start as soon as backwards from peak we get inside 3 times stddev of noise, this means from thereon we are within 99.7% of noise datapoints (normal distribution), conservative. Good


analyzer.shift_data() # start time, displacement at zero

#%% Plot all raw peak data

analyzer.plot_force_curve(x_mode="displacement", 
                          data="peak", 
                          datatype=None)

analyzer.plot_force_curve(x_mode="time", 
                          data="peak", 
                          datatype=None)




#%% Parameter screening for the Gaussian Sigma -> look at resulting forces and energies and save them individually

import numpy as np
for sigma in np.linspace(0.01, 0.1, num=10):
    analyzer.gaussian_force_displacement(
                data=None,
                sigma=sigma,
                min_points=3,
                compute_spread=True,
                plot_true=False
                )
    
    analyzer.calc_bursting_force_energy(label_add2 = f"Sigma_{sigma:.2f}",data_str = "Gauss", plot_true = False)
#    batch_analyzer.average_bursting_forces_displacements_energy(gauss_sigma = sigma)
    
#%% Energy and Force calculation based on sigma 0.04 in Gauss

analyzer.gaussian_force_displacement(
            data=None,
            sigma=0.04,
            min_points=3,
            compute_spread=True,
            plot_true=True
            )

analyzer.calc_bursting_force_energy(label_add2 = f"Sigma_{sigma:.2f}",data_str = "Gauss", plot_true = True)

#%% Export analyzer to add in batch Chlorella analysis

import pickle
from pathlib import Path

outpath = Path("saved_single_analyzers") / "Chl49_single_analyzer.pkl"
outpath.parent.mkdir(parents=True, exist_ok=True)

with open(outpath, "wb") as f:
    pickle.dump(analyzer, f)