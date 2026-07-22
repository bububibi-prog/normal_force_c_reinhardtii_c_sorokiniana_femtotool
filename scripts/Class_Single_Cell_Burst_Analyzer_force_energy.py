# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 15:19:25 2025

@author: baujulia
"""
#%% Libraries
import numpy as np
#import cv2
import matplotlib.pyplot as plt
import pandas as pd
#import trackpy as tp
from pathlib import Path
import json
#import imageio.v3 as iio
import glob
from typing import Optional, Tuple, Dict, Union
from io import StringIO
import re
from config import X_AXIS_CONFIG
import scienceplots

#%% Class
class Single_Cell_Analyzer:
    """ Class for analyzing single microalgal cell bursts measured with Femtotool """
    
    # ------------------------
    # Class-level constants
    # ------------------------
    
    REQUIRED_COLUMNS = [
        "Index", "Time [s]", "Displacement [um]",
        "Pos X [um]", "Pos Y [um]", "Pos Z [um]",
        "Force A [uN]", "Force B [uN]"
    ]
    
    REQUIRED_COLUMNS_2 = [
        "Time [s]", 
        "Pos X [um]", "Pos Y [um]", "Pos Z [um]",
        "Force A [uN]", "Force B [uN]"
    ]
    
    #%% ------------------------
    # Initialization
    # ------------------------
    
    def __init__(self,
                 filename: str,
                 base_path: Union[str, Path] = "./",
                 subfolder: str = "",
                 subfolder_cell_diameter: str = "",
                 save: bool = True,
                 plot: bool = True,
                 verbose: bool = True,
                 stiffness_meas_syst = None, # Needs to be determined!
                 stiffness_filepath =None
                 ):
        
        # Global parameters
        self.base_path = Path(base_path)
        self.save = save
        self.plot = plot
        self.verbose = verbose
        self.figsize = (5,5)
        plt.rc('font', size=20)
        plt.style.use(['science', 'no-latex'])
        self.color = plt.colormaps['viridis']
        self.start_noise: int = 0
        self.end_noise: int = 0
        self.contact_idx: int = 0
        self.peak_start_idx: int = 0
        self.cutoff_idx: int = 0
        #self.k_cell: float = 0 # Stiffness of the cell
        self.cell_diameter: float = 0 # unit um
        #self.young: float = 0
        #self.young_sd: float = 0
        #self.CC_model_RMSE_uN: float = 0
        #self.CC_model_RMSE: float = 0
        self.bursting_force: float = 0
        self.bursting_displacement: float = 0
        self.bursting_energy: float = 0
        self.energy_dw: float = 0
        self.bursting_force_per_diameter: float = 0
        self.bursting_displacement_per_diameter: float = 0
        self.bursting_energy_per_diameter: float = 0
        self.average_aquisition: float = 0
        self.std_aquisition: float =0
        self.std_slope_noise_smoothed: float = 0 # the standard deviation of noise in slope
        
        #Folder setup
        self._setup_folders()
        
        #Full path to the data file in Data/
        self.filepath = self.folders["Raw_Data"] / subfolder / filename
        self.cell_diamter_filepath = self.folders["Raw_Data"] / subfolder_cell_diameter / filename
        self.stiffness_meas_syst = stiffness_meas_syst
        self.stiffness_filepath = stiffness_filepath
        
        #Data placeholders
        self.raw_data: Optional[pd.DataFrame] = None # means either pd dataframe or None
        self.pp_despike: Optional[pd.DataFrame] = None # pp = preprocessing
        self.pp_smooth: Optional[pd.DataFrame] = None
        self.pp_slope: Optional[pd.DataFrame] = None
        self.peak_data: Optional[pd.DataFrame] = None
        self.pp_binned: Optional[pd.DataFrame] = None
        self.force_per_displacement: Optional[pd.DataFrame] = None
        self.results: Path = Path()
        self.filename: str = filename.removesuffix(".txt")
        self.baseline: Optional[np.array] = None
        self.pp_bl_corrected: Optional[pd.DataFrame] = None # Baseline corrected raw data
        self.smooth_bl_corrected: Optional[pd.DataFrame] = None # Baseline corrected smooth data
        
        self.smoothed_force_per_displacement: Optional[np.array] = None # smoothed peak force slope
        self.smoothed_peak_force: Optional[np.array] = None # smoothed peak force
        
        #Load data immediately
        self.load_data()
        
    #%% ------------------------
    # Folder setup
    # ------------------------
    
    def _setup_folders(self):
        """Create necessary folder structure."""
        self.folders = {
            "Raw_Data": self.base_path / "Raw_Data",
            "Process_Data": self.base_path / "Process_Data",
            "Process_Plots": self.base_path / "Process_Plots",
            "Result_Data": self.base_path / "Result_Data",
            "Result_Plots": self.base_path / "Result_Plots"
        }
        for name, path in self.folders.items():
            path.mkdir(parents=True, exist_ok=True)
        if self.verbose:
            print("Folder structure created or verified.")
    
    # ------------------------
    #%% Loading functions
    # ------------------------
    
    def _load_data_single_file(self, filepath: Path) -> pd.DataFrame:
        """Helper: Load a single .txt file into a DataFrame."""
        if not filepath.exists():
            raise FileNotFoundError(f"{filepath} does not exist.")
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Find the header line
        header_line_index = None
        for i, line in enumerate(lines):
            if "Pos X" in line and "Time" in line:
                header_line_index = i
                break
        if header_line_index is None:
            raise ValueError("Header line not found in the file.")
        
        # Take all lines AFTER the header
        raw_data_lines = lines[header_line_index + 1:]
        
        # Keep only lines that contain numbers (skip comments and empty lines)
        data_lines = []
        for line in raw_data_lines:
            line_clean = line.strip()
            if line_clean and not line_clean.startswith("//"):
                data_lines.append(line_clean.replace(',', '.'))  # replace decimal commas
        
        data_str = '\n'.join(data_lines)
        
        # Load into pandas
        df = pd.read_csv(StringIO(data_str), sep=r'\s+', header=None, engine="python")
        
        # Overwrite column names
        if len(df.columns) == 8:
            df.columns = self.REQUIRED_COLUMNS
            
        if len(df.columns) == 6:
            df.columns = self.REQUIRED_COLUMNS_2
            pos_z = df["Pos Z [um]"].astype(float)
            df["Displacement [um]"] = pos_z.iloc[0]-pos_z
            df["Index"] = np.arange(len(df))
            df = df[self.REQUIRED_COLUMNS]
            
        return df
    
    def _load_cell_diameter(self, filepath: Path) -> float:
        if not filepath.exists():
            raise FileNotFoundError(f"{filepath} does not exist.")
        try:
            with open(filepath, 'r') as f:
                return float(f.read().strip())
        except ValueError as e:
            raise ValueError(f"File {filepath} does not contain a valid float") from e

        
    def load_data(self):
        """Load data for this instance from self.filepath."""
        if self.verbose:
            print(f"\nLoading Data\nData is loaded from {self.filepath}")
        raw_data = self._load_data_single_file(self.filepath)
        
        stiffness_meas_syst = self._load_system_stiffness()
        
        # Correct displacement which is now displacement of measurement system plus sample 
        # with the stiffness of measurement system determined in correction with glass slide script
        if stiffness_meas_syst == None:
            self.raw_data = raw_data
            print("!!! NO MEASUREMENT SYSTEM STIFFNESS FOR CORRECTION!!!\n")
        else:
            deformation_meas_syst = raw_data["Force A [uN]"] / stiffness_meas_syst 
            deformation_sample = raw_data["Displacement [um]"] - deformation_meas_syst # total deformation minus deformation of meas syst
            raw_data["Displacement [um]"] = deformation_sample
            self.raw_data = raw_data
            self.stiffness_meas_syst = stiffness_meas_syst
            print(f"Corrected with measurement system stiffness {stiffness_meas_syst}")
        self.cell_diameter = self._load_cell_diameter(self.cell_diamter_filepath)
        if self.verbose:
            print(f"Data loaded successfully. Shape: {self.raw_data.shape}")
        
        
        
    def _load_system_stiffness(self):
        filepath = self.stiffness_filepath
        stiffness_meas_syst = self.stiffness_meas_syst
        if filepath is None:
            if stiffness_meas_syst is None:
                return None
            else:
                return stiffness_meas_syst
        elif stiffness_meas_syst is None:
        
            values = {}
            with open(filepath, "r") as f:
                for line in f:
    
                    if line.startswith("#"):
                        continue
    
                    key, value = line.strip().split("\t")
    
                    try:
                        values[key] = float(value)
                    except ValueError:
                        values[key] = value
            stiffness_meas_syst = values["stiffness_uN_per_um"]
            self.stiffness_meas_syst = stiffness_meas_syst
            return stiffness_meas_syst
        else:
            return stiffness_meas_syst
    #%% General functions
    
    def get_force_curve_data(self, data=None):
        if data is None:
            return self.raw_data, "Raw_Data"
        elif data == "peak":
            return self.peak_data, "Peak_Data"
        elif data == "binned":
            return self.pp_binned, "Binned_Peak_Data"
        elif data == "gauss":
            return self.pp_gauss, "Gauss_Peak_Data"
        else:
            return data, "Processed_Data"
    
    
    def std_dev_baseline(self,p_points = 0.7, plot_true = False):
        from scipy.stats import linregress
        time = self.raw_data["Time [s]"].values # np array
        force = self.raw_data["Force A [uN]"].values
        
        n_points = int(p_points*len(force))
        
        slope, intercept, rvalue, pvalue, stderr = linregress(
            time[:n_points],
            force[:n_points])
        
        print(f"{self.filename} Statistics fit: \n slope:{slope}, \n intercept: {intercept}, \n rvalue: {rvalue}, \n pvalue: {pvalue}, \n stderr: {stderr}\n\n")
        
        raw_force = self.raw_data["Force A [uN]"]
        baseline = slope*time + intercept
        force_corrected = raw_force - baseline
        std_dev = np.std(force_corrected[:n_points],ddof = 3)
        
        # Vizualize
        if plot_true:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0]*2, self.figsize[1]*1.08))
            
            # Left subplot: Baseline
            ax1.plot(time, 
                     raw_force,
                     color = self.color(0.0),
                     label = "Raw data")
            ax1.plot(time,baseline,
                     color = self.color(0.5),
                     label = "baseline"
                     )
            ax1.plot(time[:n_points],
            force[:n_points],
            color = self.color(0.9),
            alpha = 0.5,
            linewidth = 10,
            label = "data for lin reg")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Force ($\mu$N)")
            ax1.legend(loc='best', fontsize=10)
            ax1.set_ylim(-100,100)
            
            # Right subplot: Subtracted baseline
            ax2.plot(time, 
                     force_corrected,
                     color = self.color(0.0),
                     label = "Baseline corrected raw data")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Force ($\mu$N)")
            ax2.legend(loc='best', fontsize=10)
            ax2.set_ylim(-50,50)
            
            fig.suptitle(f"Baseline correction [{self.start_noise}:{self.end_noise}] {self.filename}")
            plt.tight_layout()
            
            outpath = self.folders["Process_Plots"] / "Baseline subtraction" / f"Baseline subtraction {self.filename}.svg"
            outpath.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(outpath, dpi = 300)
            print(f"Plot saved to {outpath}")
            plt.show()
            plt.close()
            
            plt.figure(figsize=self.figsize)
            plt.plot(time, force, color=self.color(0.0), label="Data")  # plot in $\mu$N, $\mu$m for readability
            plt.plot(time[:n_points], force[:n_points], '-', color=self.color(0.5),linewidth = 10,alpha = 0.5, label=f"region taken for std using {n_points} points")
            plt.xlabel("Time (s)")
            plt.ylabel("Force ($\mu$N)")
            plt.legend()
            plt.title(f"{self.filename} region for std")
            plt.tight_layout()
            outpath = self.folders["Result_Plots"] / "Standard deviation sensor" / f"region for std {self.filename}.svg"
            outpath.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(outpath, dpi = 300)
            print(f"Plot saved to {outpath}")
            plt.show()
            plt.close()
        
        print(f"The std dev in force is {std_dev}")
        return(std_dev)
    
    
    def frequency_data_aq(self):
        time = self.raw_data["Time [s]"].values
        dtime = time[1:len(time)] - time[0:(len(time)-1)]
        av_dtime = np.average(dtime)
        sd_dtime = np.std(dtime, ddof = 1)
        self.average_aquisition = av_dtime
        self.std_aquisition = sd_dtime
        print(f"the average aquisition for {self.filename} is 1 force point every {av_dtime} sd {sd_dtime} s")
        return(av_dtime,sd_dtime)
    
    # ------------------------
    #%% Preprocessing functions
    # ------------------------
        
    def smooth_force_time(self, med_kernel = 3, sg_window= 11, sg_order= 2, plot_true = False):
        data = self.raw_data.copy()
        time = data["Time [s]"]
        force = data["Force A [uN]"] # Apply Savitzky-Golay filter, then check the signal with overlay with and without filter
        from scipy.signal import savgol_filter, medfilt
        force_despiked = medfilt(force, kernel_size=med_kernel)
        force_smooth = savgol_filter(force_despiked, window_length=sg_window, polyorder=sg_order)
        
        self.pp_despike = data.copy()
        self.pp_despike["Force A [uN]"] = force_despiked
        
        self.pp_smooth = data.copy()
        self.pp_smooth["Force A [uN]"] = force_smooth
        
        if plot_true:
            # Create figure with 2 subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0]*2, self.figsize[1]*1.08))
            
            # Left subplot: Despiking
            ax1.plot(time, 
                     force,
                     color = self.color(0.0),
                     label = "Raw data")
            ax1.plot(time, 
                     force_despiked,
                     color = self.color(0.5),
                     label = f"Median filter {med_kernel}")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Force ($\mu$N)")
            ax1.legend(loc='best', fontsize=10)
            ax1.set_ylim(-100,100)
            
            # Right subplot: Smoothing
            ax2.plot(time, 
                     force,
                     color = self.color(0.0),
                     label = "Raw data")
            ax2.plot(time,
                     force_smooth,
                     color = self.color(0.9),
                     label = f"Savitzky-Golay window = {sg_window}, order = {sg_order}")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Force ($\mu$N)")
            ax2.legend(loc='best', fontsize=10)
            ax2.set_ylim(-100,100)
            
            fig.suptitle(f"Despike (l) & Smoothe (r) {self.filename}")
            plt.tight_layout()
            
            outpath = self.folders["Process_Plots"] / "Smooth_Despike" / f"Smooth_Despike {self.filename}.svg"
            outpath.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(outpath, dpi = 300)
            print(f"Plot saved to {outpath}")
            plt.show()
            plt.close()

    def rolling_slope(self, window, plot_true = False):
        """
        Compute rolling local slope using linear regression.
        
        Parameters
        ----------
        time : 1D array
        force : 1D array
        window : int
            Number of points per window (must be odd)
        
        Returns
        -------
        slopes : 1D array
            Local slope at each point (NaN at edges)
        """
        time = self.pp_smooth["Time [s]"].copy()
        force = self.pp_smooth["Force A [uN]"].copy()
        half = window // 2 # divide by 2 and round to lower whole number
        slopes = np.full_like(force, np.nan, dtype=float)
    
        for i in range(half, len(force) - half):
            t = time[i-half:i+half+1]
            f = force[i-half:i+half+1]
            # Linear regression: slope only
            slope = np.polyfit(t, f, 1)[0]
            slopes[i] = slope
            
        self.pp_slope = self.pp_smooth.copy()
        self.pp_slope["Force A [uN]"] = slopes
        self.pp_slope.rename(columns = {"Force A [uN]" : "Force A per time [uN/s]"}, inplace = True)
        
        if plot_true:
            # Create figure with 2 subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0]*2, self.figsize[1]*1.08))
            
            # Left subplot: Despiking
            ax1.plot(time, 
                     force,
                     color = self.color(0.0),
                     label = "Smoothed")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Force ($\mu$N)")
            ax1.legend(loc='best', fontsize=10)
            ax1.set_ylim(-100,100)
            
            # Right subplot: Smoothing
            ax2.plot(time, 
                     slopes,
                     color = self.color(0.5),
                     label = f"Slopes {window}")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Force per time ($\mu$N/s)")
            ax2.legend(loc='best', fontsize=10)
            ax2.set_ylim(-50,50)
            
            fig.suptitle(f"Slopes {self.filename}")
            plt.tight_layout()
            
            outpath = self.folders["Process_Plots"] / "Rolling_slope" / f"Slope {self.filename} window {window}.svg"
            outpath.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(outpath, dpi = 300)
            print(f"Plot saved to {outpath}")
            plt.show()
            plt.close()
        
        
    def detect_contact_slope(self, start_noise = 10, r_noise = 0.5,min_noise = 70, threshold_sigma=5, plot_true = False, early_peak = False):
        self.start_noise = start_noise
        slopes = self.pp_slope["Force A per time [uN/s]"].values
        slopes_rel = slopes[start_noise:len(slopes)]
        time = self.pp_slope["Time [s]"].copy()
        force = self.pp_smooth["Force A [uN]"].copy()
        
        # Estimate noise from early region
        end_noise = int(np.round(len(slopes)*r_noise,0))
        if end_noise < min_noise + start_noise:
            end_noise = min_noise + start_noise
            print(f"/n used minimal noise points of {min_noise}/n")
        else:
            print(f"/n used noise ratio {r_noise}/n")
            
        #Estimate noise from late region in case of early peak
        if early_peak:
            end_noise = len(slopes) - start_noise
            start_noise = len(slopes) - int(np.round(len(slopes)*r_noise,0))
            self.early_peak_slice = slice(start_noise,end_noise)
            if end_noise < min_noise + start_noise:
                end_noise = min_noise + start_noise
                print(f"/n used minimal noise points of {min_noise} starting from end/n")
            else:
                print(f"/n used noise ratio {r_noise} starting from end/n")
        
        self.end_noise = end_noise
        slope_noise = slopes[~np.isnan(slopes)][start_noise:end_noise]
        self.std_slope_noise_smoothed = np.std(slope_noise)
        threshold = threshold_sigma * self.std_slope_noise_smoothed # Threshold as a factor of the slope std
        
        contact_idx = np.argmax(np.abs(slopes_rel) > threshold) + self.start_noise
        self.contact_idx = contact_idx
        contact = np.zeros(len(slopes))
        contact.fill(-1000)
        contact[contact_idx] = 1000
        
        if plot_true:
            # Create figure with 2 subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0]*2, self.figsize[1]*1.08))
            
            # Left subplot: Despiking
            ax1.plot(time, 
                     force,
                     color = self.color(0.0),
                     label = "Smoothed")
            ax1.plot(time,contact,
                     color = self.color(0.9),
                     label = "contact"
                     
                     )
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Force ($\mu$N)")
            ax1.legend(loc='best', fontsize=10)
            ax1.set_ylim(-100,100)
            
            # Right subplot: Smoothing
            ax2.plot(time, 
                     slopes,
                     color = self.color(0.5),
                     label = f"Slopes")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Force per time ($\mu$N/s)")
            ax2.legend(loc='best', fontsize=10)
            ax2.set_ylim(-50,50)
            
            fig.suptitle(f"Slopes {self.filename}")
            plt.tight_layout()
            
            outpath = self.folders["Process_Plots"] / "Peak detection" / f"Peak {self.filename}.svg"
            outpath.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(outpath, dpi = 300)
            print(f"Plot saved to {outpath}")
            plt.show()
            plt.close()
        
        #return contact_idx, slopes, threshold

    def subtract_baseline(self,margin = 10, plot_true = False, early_peak = False):
        
        """
        Linear regression done on smoothed data
        Baseline subtraction from raw data
        Used larger region for baseline subtraction than for peak identification
        """
        self.margin = margin
        from scipy.stats import linregress
        time = self.pp_smooth["Time [s]"].values # np array
        force = self.pp_smooth["Force A [uN]"].values
        pre_contact_slice = slice(self.start_noise,self.contact_idx-margin)
        
        slope, intercept, rvalue, pvalue, stderr = linregress(
            time[pre_contact_slice],
            force[pre_contact_slice])
        
        if early_peak:
            slope, intercept, rvalue, pvalue, stderr = linregress(
                time[self.early_peak_slice],
                force[self.early_peak_slice])
        
        print(f"{self.filename} Statistics fit: \n slope:{slope}, \n intercept: {intercept}, \n rvalue: {rvalue}, \n pvalue: {pvalue}, \n stderr: {stderr}\n\n")
        
        raw_force = self.raw_data["Force A [uN]"]
        smooth_force = self.pp_smooth["Force A [uN]"]
        baseline = slope*time + intercept
        self.baseline = baseline
        force_corrected = raw_force - baseline
        smooth_force_corrected = smooth_force - baseline
        self.pp_bl_corrected = self.pp_smooth.copy()
        self.pp_bl_corrected["Force A [uN]"] = force_corrected
        self.smooth_bl_corrected = self.pp_smooth.copy()
        self.smooth_bl_corrected["Force A [uN]"] = smooth_force_corrected
        
        # Vizualize
        if plot_true:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0]*2, self.figsize[1]*1.08))
            
            # Left subplot: Baseline
            ax1.plot(time, 
                     raw_force,
                     color = self.color(0.0),
                     label = "Raw data")
            ax1.plot(time,baseline,
                     color = self.color(0.5),
                     label = "baseline"
                     )
            if not early_peak:
                ax1.plot(time[pre_contact_slice],
                force[pre_contact_slice],
                color = self.color(0.9),
                alpha = 0.5,
                linewidth = 10,
                label = "data for lin reg")
                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Force ($\mu$N)")
                ax1.legend(loc='best', fontsize=10)
                ax1.set_ylim(-100,100)
            
            if early_peak:
                ax1.plot(time[self.early_peak_slice],
                force[self.early_peak_slice],
                color = self.color(0.9),
                alpha = 0.5,
                linewidth = 10,
                label = "data for lin reg")
                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Force ($\mu$N)")
                ax1.legend(loc='best', fontsize=10)
                ax1.set_ylim(-100,100)
            
            # Right subplot: Subtracted baseline
            ax2.plot(time, 
                     force_corrected,
                     color = self.color(0.0),
                     label = "Baseline corrected raw data")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Force ($\mu$N)")
            ax2.legend(loc='best', fontsize=10)
            ax2.set_ylim(-50,50)
            
            fig.suptitle(f"Baseline correction [{self.start_noise}:{self.end_noise}] {self.filename}")
            plt.tight_layout()
            
            outpath = self.folders["Process_Plots"] / "Baseline subtraction" / f"Baseline subtraction {self.filename}.svg"
            outpath.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(outpath, dpi = 300)
            print(f"Plot saved to {outpath}")
            plt.show()
            plt.close()
            
            
            
    def _true_peak_start(self,factor_stddev, plot_true):
        """
        The contact index is at 5x the standard deviation of slope in baseline region. 
        We define the start of the peak at the first slope value that falls within (x times) that standard deviation # changed to 5x again!
        """
        force = self.pp_bl_corrected["Force A [uN]"]
        slopes = self.pp_slope["Force A per time [uN/s]"].values
        displacement = self.pp_slope["Displacement [um]"].values
        time = self.pp_slope["Time [s]"].values
        threshold = self.std_slope_noise_smoothed*factor_stddev
        i = self.contact_idx
        # Move backwards from contact_idx
        
        if i == 0:
            raise ValueError("All slopes before contact index exceed threshold.")
        
        while i > 0:
            # Check if force is within baseline noise range
            if slopes[i] <= threshold:
                
                true_peak_start = np.zeros(len(slopes))
                true_peak_start.fill(-1000)
                true_peak_start[i] = 1000
                
                if plot_true:
                    plt.figure(figsize=self.figsize)
                    plt.plot(time, slopes,color = self.color(0.0), label = "Slope smoothed")
                    plt.plot(time, true_peak_start, color=self.color(0.5), label = "Peak start")
                    plt.xlabel("Time (s)")
                    plt.ylabel("Slope ($\mu$N/s)")
                    plt.title(f"{self.filename}")
                    plt.ylim(-10,50)
                    plt.tight_layout()
                    outpath = self.folders["Process_Plots"] / "Peak_true_start" / f"peak_start_slope_time{self.filename}.svg"
                    outpath.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(outpath, dpi = 300)
                    plt.show()
                    plt.close()
                    
                    plt.figure(figsize=self.figsize)
                    plt.plot(displacement, force,color = self.color(0.0), label = "BL corrected raw")
                    plt.plot(displacement, true_peak_start, color=self.color(0.5), label = "Peak start")
                    plt.xlabel("Displacement ($\mu$m)")
                    plt.ylabel("Force ($\mu$N)")
                    plt.title(f"{self.filename}")
                    plt.ylim(-10,50)
                    plt.tight_layout()
                    outpath = self.folders["Process_Plots"] / "Peak_true_start" / f"peak_start_displacement_force{self.filename}.svg"
                    outpath.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(outpath, dpi = 300)
                    plt.show()
                    plt.close()
                
                return i
            i -= 1

        # If nothing found, return 0
        return 0
            
    
    def cutoff2(self, min_fraction=0.7,drop_limit = 5, factor_stddev = 1, plot_true=False):
        """
        Select the peak data in baseline-corrected force.
        Dynamically grow the peak region starting from the first detected peak until
        the newest point falls below min_fraction of the current local maximum.
        """
        from scipy.signal import find_peaks
        import numpy as np
    
        time = self.pp_bl_corrected["Time [s]"].values 
        force = self.pp_bl_corrected["Force A [uN]"].values
    
        # Compute baseline noise
        noise_std = np.std(force[:self.contact_idx])
        prominence_levels = noise_std * np.array([6, 5, 4, 3, 2])
    
        # Search for first peak after contact
        search_force = force[self.contact_idx:]
        
        
        first_peak_idx = None
    
        for prom in prominence_levels:
            peaks, _ = find_peaks(search_force, prominence=prom) # Peak indices, relative to search force -> add self.contact_idx
            if len(peaks) == 0:
                continue
        
            peak_region_start = peaks[0] # Pick the first peak, index of the first peak, relative to search force.
            peak_region_end = peak_region_start + 1
            local_max = search_force[peak_region_start]
            last_good_idx = peak_region_start
            drop_counter = 0
        
            while peak_region_end < len(search_force):
                val = search_force[peak_region_end]
                peak_region_end += 1
        
                if val >= min_fraction * local_max:
                    local_max = max(local_max, val)
                    last_good_idx = peak_region_end - 1
                    drop_counter = 0
                else:
                    drop_counter += 1
                    if drop_counter >= drop_limit:
                        break
                
            region_force = search_force[:last_good_idx +1]
            max_force = np.max(region_force)
            within_80 = np.where(region_force >= 0.8 * max_force)[0] # Due to noisy data take last index within 90% of maximal force
            first_peak_idx = self.contact_idx + within_80[-1]
            break
        
        if first_peak_idx is None:
            raise ValueError("No valid peak found.")

        self.cutoff_idx = first_peak_idx

        # crop peak data
        # Find true peak start (within noise std before contact index)
        peak_start_idx = self._true_peak_start(factor_stddev, plot_true) # Set True to see slope etc plots for peak start finding
        self.peak_start_idx = peak_start_idx

        peak = self.pp_bl_corrected.iloc[peak_start_idx:first_peak_idx+1].copy() 
        self.peak_data = peak
        time_peak = peak["Time [s]"]
        force_peak = peak["Force A [uN]"]
        
        
        # Vizualize
        if plot_true:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0]*2, self.figsize[1]*1.08))
            
            # Left subplot: cutoff and start
            ax1.plot(time, 
                     force,
                     color = self.color(0.0),
                     label = "Baseline corrected data")
            ax1.plot(time_peak, 
                     force_peak,
                     color = self.color(0.9),
                     label = "peak selected",
                     alpha = 0.8#,
                     #linewidth = 2
                     )
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Force ($\mu$N)")
            ax1.legend(loc='best', fontsize=10)
            ax1.set_ylim(-10,300)
            
            # Right subplot: Subtracted baseline
            ax2.plot(time_peak, 
                     force_peak,
                     color = self.color(0.0),
                     label = "Selected peak")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Force ($\mu$N)")
            ax2.legend(loc='best', fontsize=10)
            ax2.set_ylim(-10,300)
            
            fig.suptitle(f"Selected peak: {self.filename}")
            plt.tight_layout()
            
            outpath = self.folders["Process_Plots"] / "Peak selected" / f"Peak selected {self.filename}.svg"
            outpath.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(outpath, dpi = 300)
            print(f"Plot saved to {outpath}")
            plt.show()
            plt.close()

    def check_normality_slopes_baseline(self,early_peak = False):
        force_slope = self.pp_slope["Force A per time [uN/s]"].values
        
        # Check normality of noise
        
        if early_peak:
            force_slope_noise = force_slope[self.early_peak_slice]
        else:
            force_slope_noise = force_slope[slice(self.start_noise,self.contact_idx-self.margin)]
        
        # QQ plot:
        import scipy.stats as stats
        plt.figure(figsize =(5,5)) 
        stats.probplot(force_slope_noise, dist="norm", plot=plt)
        plt.title(f"QQ Force {self.filename}")
        outpath = self.folders["Process_Plots"] / "Normality_Slope_Noise_QQ" / f"QQ_plot {self.filename}.svg"
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi = 300)
        print(f"Plot saved to {outpath}")
        plt.show()
        plt.close()
            

    def shift_data(self):
        data = self.peak_data.copy()

        data["Time [s]"] = (data["Time [s]"] - data["Time [s]"].iloc[0])
        data["Displacement [um]"] = (data["Displacement [um]"] - data["Displacement [um]"].iloc[0])
        self.peak_data = data
        
        
    def bin_force_displacement(self, data=None, bin_width=0.05, 
                          method="median", min_points=3, 
                          compute_spread=True, plot_true=False):
        """
        Bin force values by displacement to obtain one force per displacement.
        
        Parameters
        ----------
        data : DataFrame or None
            Input dataset. If None → self.pp_bl_corrected is used.
        bin_width : float
            Width of displacement bins in µm (default = 0.05 µm).
        method : str
            "median" (recommended) or "mean".
        min_points : int
            Minimum number of points required per bin.
        compute_spread : bool
            If True, compute std and IQR per bin.
        plot_true : bool
            Plot raw vs binned data.
        """
        # ------------------------
        # Get data
        # ------------------------
        if data is None:
            data = self.peak_data.copy()
            
        elif data == "gauss":
            data = self.pp_gauss
            
        elif data == "peak":
            data = self.peak_data
            
        else:
            raise ValueError("Invalid data entered. Enter gauss, or peak to get respective data")
    
        displacement = data["Displacement [um]"].values
        force = data["Force A [uN]"].values
        time = data["Time [s]"].values
    
        # ------------------------
        # Define bins
        # ------------------------
        d_min = np.min(displacement)
        d_max = np.max(displacement)
        bins = np.arange(d_min, d_max + bin_width, bin_width)
    
        bin_indices = np.digitize(displacement, bins)
    
        # ------------------------
        # Aggregate per bin
        # ------------------------
        binned_time = []
        binned_disp = []
        binned_force = []
        binned_std = []
        binned_iqr = []
        counts = []
    
        for b in range(1, len(bins)):
            mask = bin_indices == b
    
            if np.sum(mask) < min_points:
                continue
            
            t_vals = time[mask]
            #d_vals = displacement[mask]
            f_vals = force[mask]
            
            #Representative time
            t_rep = np.mean(t_vals)
            # Representative displacement
            d_rep = bins[b-1] + bin_width / 2
    
            # Representative force
            if method == "median":
                f_rep = np.median(f_vals)
            elif method == "mean":
                f_rep = np.mean(f_vals)
            elif method == "max":
                f_rep = np.max(f_vals)
            else:
                raise ValueError("method must be 'median' or 'mean'")
    
            binned_time.append(t_rep)
            binned_disp.append(d_rep)
            binned_force.append(f_rep)
            counts.append(len(f_vals))
    
            if compute_spread:
                binned_std.append(np.std(f_vals, ddof=1))
                q75, q25 = np.percentile(f_vals, [75, 25])
                binned_iqr.append(q75 - q25)
    
        # ------------------------
        # Store result
        # ------------------------
        df_binned = pd.DataFrame({
            "Time [s]": binned_time,
            "Displacement [um]": binned_disp,
            "Force A [uN]": binned_force,
            "Counts": counts
        })
        
        
    
        if compute_spread:
            df_binned["Force STD [uN]"] = binned_std
            df_binned["Force IQR [uN]"] = binned_iqr
    
        self.pp_binned = df_binned
    
        # ------------------------
        # Plot
        # ------------------------
        if plot_true:
            plt.figure(figsize=self.figsize)
            # Raw data
            plt.plot(displacement, force, '.', 
                     color=self.color(0.0), alpha=0.3, label="Raw baseline corrected")
    
            # Binned
            plt.plot(df_binned["Displacement [um]"],
                     df_binned["Force A [uN]"],
                     'o', color=self.color(0.8),
                     label=f"Binned ({method})")
    
            plt.xlabel("Displacement ($\\mu$m)")
            plt.ylabel("Force ($\\mu$N)")
            plt.title(f"Binned force-displacement {self.filename}")
            plt.legend()
            plt.tight_layout()
    
            outpath = self.folders["Process_Plots"] / "Binning" / f"Binned_{self.filename}.svg"
            outpath.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(outpath, dpi=300)
            print(f"Plot saved to {outpath}")
    
            plt.show()
            plt.close()
            
    
        return df_binned
    
    
    
    
    def gaussian_force_displacement(
            self,
            data=None,
            sigma=0.1,
            min_points=3,
            compute_spread=True,
            plot_true=False
            ):
        """
        Gaussian-smoothed force-displacement curve (alternative to binning).
    
        Parameters
        ----------
        data : DataFrame or None
            Input dataset. If None → self.peak_data is used.
        sigma : float
            Gaussian smoothing strength in index space (NOT physical units).
            Typical range: 0.5–2 for ~20–50 points.
        min_points : int
            Minimum number of raw points required (safety check only).
        compute_spread : bool
            Computes local rolling std as an uncertainty proxy.
        plot_true : bool
            Plot raw vs smoothed curve.
        """
    
        from scipy.ndimage import gaussian_filter1d
    
        # ------------------------
        # Get data
        # ------------------------
        if data is None:
            data = self.peak_data.copy()
    
        displacement = data["Displacement [um]"].values
        force = data["Force A [uN]"].values
        time = data["Time [s]"].values
    
        # ------------------------
        # Ensure sorting by displacement (critical!)
        # ------------------------
        sort_idx = np.argsort(displacement)
        displacement = displacement[sort_idx]
        force = force[sort_idx]
        time = time[sort_idx]
    
        # ------------------------
        # Optional sanity check
        # ------------------------
        if len(force) < min_points:
            raise ValueError("Not enough points for Gaussian smoothing")
    
        # ------------------------
        # Gaussian smoothing
        # ------------------------
        
        median_spacing = np.median(np.diff(displacement))  # µm per point
        #print(f"Median spacing between raw datapoints: {median_spacing}")
        if median_spacing == 0:
            raise ValueError("Two identical displacement values in sequence cannot calculate sigma in points")
        sigma_pt = sigma / median_spacing  # points
        
        # Smooth the padded array with mode='nearest' (constant at the new edges)
        force_smooth = gaussian_filter1d(force, sigma=sigma_pt, mode = 'reflect') 
        
        # ------------------------
        # Build "binned-like" dataframe
        # (keeps compatibility with your pipeline)
        # ------------------------
        df_gauss = pd.DataFrame({
            "Time [s]": time,
            "Displacement [um]": displacement,
            "Force A [uN]": force_smooth,
            "Force RAW [uN]": force
        })
        
        # ------------------------
        # Optional local spread estimate (rolling std)
        # ------------------------
        if compute_spread:
            window = max(3, int(3 * sigma_pt))
            rolling_std = (
                pd.Series(force)
                .rolling(window=window, center=True)
                .std()
                .to_numpy()
            )
            df_gauss["Force STD [uN]"] = rolling_std
    
        # ------------------------
        # Store in class (IMPORTANT: same variable name)
        # ------------------------
        self.pp_gauss = df_gauss  # keep compatibility with downstream code
        self.gaussian_sigma = np.round(sigma,2)
    
        # ------------------------
        # Plot
        # ------------------------
        if plot_true:
            plt.figure(figsize=self.figsize)
    
            plt.plot(displacement, force,
                     '.', color=self.color(0.0), label="Raw")
    
            plt.plot(displacement, force_smooth,
                     '-', color=self.color(0.5),
                     label=f"Gaussian $σ$={sigma:.2f}")
    
            plt.xlabel("Displacement ($\\mu$m)")
            plt.ylabel("Force ($\\mu$N)")
            plt.title(f"Gaussian smoothed force-displacement {self.filename}")
            plt.legend()
            plt.tight_layout()
    
            outpath = self.folders["Process_Plots"] / "Gaussian" / f"Gaussian_{self.filename}_Sigma_{sigma:.2f}.svg"
            outpath.parent.mkdir(parents=True, exist_ok=True)
    
            plt.savefig(outpath, dpi=300)
            print(f"Plot saved to {outpath}")
    
            plt.show()
            plt.close()
    
        return df_gauss

#%% Energy and force to burst


    def calc_bursting_force_energy(self,label_add2 = "", data_str = None, plot_true = False):
        if data_str == "Gauss":
            data = self.pp_gauss.copy()
            label_add = f"Gauss sigma {self.gaussian_sigma}"
            
        elif data_str == "Binned":
            data = self.pp_binned.copy()
            label_add = "Binned"
        else:
            data = self.peak_data
            label_add = " raw peak"
            
        force = data["Force A [uN]"].values 
        displacement = data["Displacement [um]"].values
        data_raw = self.peak_data.copy()
        force_raw = data_raw["Force A [uN]"].values
        displacement_raw = data_raw["Displacement [um]"].values
        
        if plot_true:
            plt.figure(figsize=self.figsize)
            plt.plot(displacement, force,color = self.color(0.0), label = "Calc based on " + label_add)
            plt.plot(displacement_raw,force_raw, '.', color = self.color(0.5),alpha = 0.5, label = "Raw")
            plt.xlabel("Displacement ($\mu$m)")
            plt.ylabel("Residual Force ($\mu$N)")
            plt.title(f"Smooth for bursting force {self.filename}")
            plt.legend()
            plt.tight_layout()
            outpath = self.folders["Result_Plots"] / "Bursting force and energy" / f"{data_str}_burst_force_energy_{self.filename}_{label_add2}.svg"
            outpath.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(outpath, dpi = 300)
            plt.show()
            plt.close()
        
        bursting_force = np.max(force)
        bursting_index = np.argmax(force)
        bursting_displacement = displacement[bursting_index]
        energy_burst = np.trapz(force,displacement) # pJ
        
        volume_cell = 4/3*np.pi*(self.cell_diameter/2)**3 * 1e-18 # m^3
        print(f"Volume: {volume_cell}\n")
        weight_cell = volume_cell*998 # kg with water density
        print(f"mass cell wet: {weight_cell}\n")
        dry_weight_cell = weight_cell*0.1 # Lee et al (2013)
        print(f"mass cell dry: {dry_weight_cell}\n")
        energy_per_dw = energy_burst*1e-12 / dry_weight_cell/1000 # kJ/kg
        print(f"energy per DW: {energy_per_dw} kJ/kg")
        
        self.bursting_force = bursting_force
        self.bursting_displacement = bursting_displacement
        self.bursting_energy = energy_burst
        self.energy_dw = energy_per_dw
        
        # Check the influence of varying diameter.
        self.bursting_force_per_diameter = bursting_force/self.cell_diameter
        self.bursting_displacement_per_diameter = bursting_displacement/self.cell_diameter
        self.bursting_energy_per_diameter = energy_burst/self.cell_diameter
        
        print(f"Bursting force of {self.filename} is {bursting_force} uN\n Bursting energy is {energy_burst} pJ (10^-12 J)\n Bursting displacement is {bursting_displacement}")
        return(bursting_force,energy_burst,bursting_displacement)
    
    
    # ------------------------
    #%% Plotting functions
    # ------------------------
    
    
    def plot_force_curve(self, x_mode="displacement", data=None, datatype=None):
        """
        Plot force vs displacement or force vs time for a single file.
        """
        if x_mode not in X_AXIS_CONFIG:
            raise ValueError(f"x_mode must be one of {list(X_AXIS_CONFIG.keys())}")
    
        x_cfg = X_AXIS_CONFIG[x_mode]
    
        if data is None:
            data = self.raw_data
            datatype = "Raw_Data"
            
        elif data == "raw":
            data = self.raw_data
            datatype = "Raw_Data"
            
        elif data == "peak":
            data = self.peak_data
            datatype = "Peak_Data"
    
        if self.verbose:
            print(f"\nPlotting Force vs {x_cfg['tag']} of {datatype} {self.filename}")
    
        plt.figure(figsize=self.figsize)
    
        plt.plot(
            data[x_cfg["col"]], 
            data["Force A [uN]"],
            '.',
            color=self.color(0.0)
        )
    
        plt.title(f"{datatype} {self.filename}")
        plt.xlabel(x_cfg["label"])
        plt.ylabel("Force ($\mu$N)")
    
        outpath = (
            self.folders["Process_Plots"]
            / f"{datatype}"
            / f"Force_vs_{x_cfg['tag']} {self.filename}.svg"
        )
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=300)
        print(f"Plot saved to {outpath}")
    
        plt.show()
        plt.close()
