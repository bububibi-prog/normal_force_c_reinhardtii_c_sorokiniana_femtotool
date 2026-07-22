# -*- coding: utf-8 -*-
"""
Batch Analyzer for Single Cell Burst Analysis

@author: baujulia
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union
import json
from Class_Single_Cell_Burst_Analyzer_force_energy import Single_Cell_Analyzer
from config import X_AXIS_CONFIG
import scipy.stats as stats
import scienceplots


class BurstBatchAnalyzer:
    """
    Batch analyzer that applies Single_Cell_Analyzer to all files in a folder.
    """
    
    def __init__(self,
                 base_path: Union[str, Path] = "./",
                 subfolder: str = "",
                 subfolder_cell_diameter: str = "",
                 file_pattern: str = "*.txt",
                 save: bool = True,
                 plot: bool = True,
                 verbose: bool = True,
                 stiffness_filepath =None,
                 stiffness_meas_syst = None):
        """
        Initialize the batch analyzer.
        
        Parameters
        ----------
        base_path : str or Path
            Base directory containing Raw_Data folder
        subfolder : str
            Subfolder within Raw_Data (e.g., "Femto")
        file_pattern : str
            Glob pattern for finding files (default: "*.txt")
        save : bool
            Whether to save outputs
        plot : bool
            Whether to generate plots
        verbose : bool
            Whether to print progress information
        """
        self.base_path = Path(base_path)
        self.subfolder = subfolder
        self.subfolder_cell_diameter = subfolder_cell_diameter
        self.file_pattern = file_pattern
        self.save = save
        self.plot = plot
        self.verbose = verbose
        self.stiffness_meas_syst_path = stiffness_filepath
        self.stiffness_meas_syst = stiffness_meas_syst
        
        # Storage for all analyzers
        self.analyzers: Dict[str, Single_Cell_Analyzer] = {}
        self.file_list: List[Path] = []
        
        # Batch results
        self.av_cell_diameter: float = 0 
        self.sd_cell_diameter: float =0
        
        self.av_bursting_force: float = 0
        self.sd_bursting_force: float = 0
        
        self.av_bursting_energy : float = 0
        self.sd_bursting_energy : float = 0
        
        self.av_bursting_displacement : float = 0
        self.sd_bursting_displacement : float = 0
        
        self.av_bursting_force_per_diameter : float = 0
        self.sd_bursting_force_per_diameter : float = 0
        
        self.av_bursting_energy_per_diameter : float = 0
        self.sd_bursting_energy_per_diameter : float = 0
        
        self.av_bursting_displacement_per_diameter : float = 0
        self.sd_bursting_displacement_per_diameter : float = 0
        
        self.av_energy_per_dw : float = 0
        self.sd_energy_per_dw : float = 0
        
        
        self.batch_results: Optional[pd.DataFrame] = None
        
        # Setup folders
        self._setup_batch_folders()
        
        # Discover files
        self._discover_files()
        
        # Plotting
        self.figsize = (5,5)
        plt.rc('font', size=20)
        plt.style.use(['science', 'no-latex'])
        self.color = plt.colormaps['viridis']
        self.color_overlays: np.array = None
        
    #%% Setup
    
    def _set_overlay_colors(self):
        self.color_overlays = plt.cm.viridis(np.linspace(0, 0.9, len(self.analyzers)))
    
    
    def _setup_batch_folders(self):
        """Create batch-specific folders."""
        self.batch_folders = {
            "Batch_Results": self.base_path / "Batch_Results",
            "Batch_Plots": self.base_path / "Batch_Plots"
        }
        for name, path in self.batch_folders.items():
            path.mkdir(parents=True, exist_ok=True)
        if self.verbose:
            print("Batch folder structure created or verified.")
            
            
    def _discover_files(self):
        """Find all files matching the pattern in Raw_Data folder."""
        raw_data_path = self.base_path / "Raw_Data" / self.subfolder
        raw_data_path_cell_diameter = self.base_path / "Raw_Data" / self.subfolder_cell_diameter
        self.file_list = sorted(raw_data_path.glob(self.file_pattern))
        self.file_list_cell_diameter = sorted(raw_data_path_cell_diameter.glob(self.file_pattern))
        
        if self.verbose:
            print(f"\nDiscovered {len(self.file_list)} files in {raw_data_path}")
            for f in self.file_list:
                print(f"  - {f.name}")
                
    def _load_system_stiffness(self):
        filepath = self.stiffness_meas_syst_path
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
    
    def load_all_data(self):
        """Load data for all discovered files."""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Loading all data files...")
            print(f"{'='*60}")
        
        raw_data_path = self.base_path / "Raw_Data" / self.subfolder
        
        stiffness_meas_syst = self._load_system_stiffness()
        
        for filepath in self.file_list:
            filename = str(filepath.relative_to(raw_data_path))
            #filename = filepath
            
            if self.verbose:
                print(f"\nProcessing: {filename}")
            
            try:
                analyzer = Single_Cell_Analyzer(
                    filename=filename,
                    subfolder = self.subfolder,
                    subfolder_cell_diameter=self.subfolder_cell_diameter,
                    base_path=self.base_path,
                    save=self.save,
                    plot=False,  # Suppress individual plots during batch load
                    verbose=False,# Suppress individual verbose output
                    stiffness_meas_syst = stiffness_meas_syst # To be determined with only glass slide
                )
                # Store with just the filename for easier access
                self.analyzers[filepath.name] = analyzer
                
                if self.verbose:
                    print(f"  ✓ Loaded successfully")
                    
            except Exception as e:
                print(f"  ✗ Error loading {filename}: {str(e)}")
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Successfully loaded {len(self.analyzers)}/{len(self.file_list)} files")
            print(f"{'='*60}\n")
        # set colors for overlay plots
        self._set_overlay_colors()
            
    def get_analyzer(self, filename: str) -> Optional[Single_Cell_Analyzer]:
        """
        Get a specific analyzer by filename.
        
        Parameters
        ----------
        filename : str
            Name of the file (with or without .txt extension)
        
        Returns
        -------
        Single_Cell_Analyzer or None
        """
        if not filename.endswith('.txt'):
            filename = filename + '.txt'
        
        return self.analyzers.get(filename)
    
    def __len__(self):
        """Return number of analyzers loaded."""
        return len(self.analyzers)
    
    def __getitem__(self, key: Union[str, int]) -> Single_Cell_Analyzer:
        """
        Access analyzer by filename (str) or index (int).
        """
        if isinstance(key, int):
            filename = list(self.analyzers.keys())[key]
            return self.analyzers[filename]
        else:
            return self.get_analyzer(key)
            
    
    #%% Plotting
    
    def plot_all_force_curves(
        self,
        overlay: bool = False,
        x_mode: str = "displacement",
        data=None,
        datatype=None,
        x_low = 0,
        x_high = 10,
        y_low = 0,
        y_high = 100
    ):
        """
        Plot force curves (Force vs displacement or time) for all samples.
        """
        if not self.analyzers:
            print("No data loaded. Run load_all_data() first.")
            return
    
        if x_mode not in X_AXIS_CONFIG:
            raise ValueError(f"x_mode must be one of {list(X_AXIS_CONFIG.keys())}")
    
        x_cfg = X_AXIS_CONFIG[x_mode]
    
        if overlay:
            self._plot_overlay(
                plot_type=f"Force_vs_{x_cfg['tag']}",
                x_col=x_cfg["col"],
                y_col="Force A [uN]",
                xlabel=x_cfg["label"],
                ylabel="Force ($\mu$N)",
                data=data,
                datatype=datatype,
                x_low = x_low,
                x_high = x_high,
                y_low = y_low,
                y_high = y_high
            )
        else:
            for analyzer in self.analyzers.values():
                analyzer.plot_force_curve(
                    x_mode=x_mode,
                    data=data,
                    datatype=datatype,
                    
                )
                
    def _plot_overlay(
        self,
        plot_type: str,
        x_col: str,
        y_col: str,
        xlabel: str,
        ylabel: str,
        data=None,
        datatype=None,
        x_low = 0,
        x_high = 10,
        y_low = 0,
        y_high = 10
    ):
        """
        Helper function to create overlay plots.
        """
        if self.verbose:
            print(f"\nCreating overlay plot: {plot_type}")
    
        # Use constrained layout instead of tight_layout
        fig, ax = plt.subplots(figsize=self.figsize, constrained_layout=True)
        
        self._set_overlay_colors()
        colors = self.color_overlays
    
        for idx, (name, analyzer) in enumerate(self.analyzers.items()):
    
            plot_data, default_type = analyzer.get_force_curve_data(data)
    
            label_type = datatype if datatype is not None else default_type
            label = f"{analyzer.filename} ({label_type})"
    
            ax.plot(
                plot_data[x_col],
                plot_data[y_col],
                color=colors[idx],
                alpha=0.7,
                #label=label
            )
    
        # Labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(x_low,x_high)
        ax.set_ylim(y_low,y_high)
        ax.set_title(f"Batch Analysis: {plot_type}")
    
        # Force square plot area
        ax.set_box_aspect(1)
    
        # Legend outside WITHOUT shrinking axes
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=8 
        )
    
        # Save figure
        if self.save:
            outpath = self.batch_folders["Batch_Plots"] / f"Batch_{plot_type}_{datatype}_Overlay.svg"
            fig.savefig(outpath, dpi=300, bbox_inches="tight")
            if self.verbose:
                print(f"Overlay plot saved to {outpath}")
    
        plt.show()
        plt.close(fig)
                
                

    def frequency_data_aq_all(self):
        aq_all = []
        sd_all = []
        for name, analyzer in self.analyzers.items():
            aq,sd_aq = analyzer.frequency_data_aq()
            aq_all.append(aq)
            sd_all.append(sd_aq)
            
        overall_av_aq = np.average(aq_all)
        overall_sd_aq = np.sqrt(np.std(aq_all,ddof = 1)**2 + np.average(sd_all)**2)
        print(f"The overall aquisition is 1 datapoint every {overall_av_aq} sd {overall_sd_aq} s.")
        self.aquisition = overall_av_aq
        
        
                
    def plot_slopes_overlay(self,datatype = "gauss"):
        
        plt.figure(figsize=self.figsize)
        self._set_overlay_colors()
        colors = self.color_overlays
        
        #%%1st plot Stiffness all
        for i, (name, analyzer) in enumerate(self.analyzers.items()):
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
            smoothed_force_per_displacement = np.diff(data["Force A [uN]"].values)/np.diff(data["Displacement [um]"].values)
            
            if analyzer.cell_diameter == 0:
                print(f"Analyzer {name} has cell diameter 0 import valid cell diameter\n")
                break
            
            rel_indent = displacement/(analyzer.cell_diameter/2)
            rel_indent_below_10_perc = rel_indent[rel_indent<= 0.1]
            displacement_below_10_perc = displacement[rel_indent<= 0.1]
            force_per_displacement_below_10_perc = smoothed_force_per_displacement[rel_indent<= 0.1]
            
            # 1st plot
            plt.plot(
                rel_indent_below_10_perc,
                force_per_displacement_below_10_perc,
                color=colors[i],
                label=name
                )
            plt.plot(
                rel_indent,
                smoothed_force_per_displacement,'--',
                color=colors[i],
                
                )
        
        plt.xlabel("Rel. indentation")
        plt.ylabel("Stiffness ($\mu$N/$\mu$m)")
        plt.title("Smoothed slope")
        plt.ylim(-2,27)
        plt.xlim(0,1.8)
        #plt.legend(
        #loc="center left",
        #bbox_to_anchor=(1.02, 0.5),
        #frameon=False)
        plt.tight_layout()
        
        outpath = self.batch_folders["Batch_Plots"] / "Smoothed_Slope_Overlay.svg"
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        
        #%%2nd plot
        plt.figure(figsize=self.figsize)
        for i, (name, analyzer) in enumerate(self.analyzers.items()):
            
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
            smoothed_peak_force = analyzer.smoothed_peak_force
            
            rel_indent = displacement/(analyzer.cell_diameter/2) #(Lacorre2024)
            rel_indent_below_10_perc = rel_indent[rel_indent<= 0.1]
            displacement_below_10_perc = displacement[rel_indent<= 0.1]
            force_below_10_perc = smoothed_peak_force[rel_indent<=0.1]
            
            
            plt.plot(
                rel_indent_below_10_perc,
                force_below_10_perc,
                color=colors[i],
                label=name
                )
            plt.plot(
                rel_indent,
                smoothed_peak_force,'--',
                color=colors[i],
                )
        
        plt.xlabel("Rel. indentation")
        plt.ylabel("Force ($\mu$N)")
        plt.title("Smoothed force")
        plt.ylim(-2,30)
        plt.xlim(0,2)
        plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False)
        plt.tight_layout()
        
        outpath = self.batch_folders["Batch_Plots"] / "Smoothed_Force_Overlay.svg"
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        
        #%%3rd plot
        plt.figure(figsize=self.figsize)
        for i, (name, analyzer) in enumerate(self.analyzers.items()):
            
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
            smoothed_peak_force = data["Force A [uN]"].values
            
            rel_indent = displacement/(analyzer.cell_diameter/2) #(Lacorre2024)
            rel_indent_below_10_perc = rel_indent[rel_indent<= 0.1]
            displacement_below_10_perc = displacement[rel_indent<= 0.1]
            force_below_10_perc = smoothed_peak_force[rel_indent<=0.1]
            x_axis_model = np.logspace(-3,1,100)
            
            plt.plot(
                rel_indent_below_10_perc,
                force_below_10_perc,
                color=self.color_overlays[i],
                label=name
                )
            plt.plot(
                rel_indent,
                smoothed_peak_force,'--',
                color=self.color_overlays[i],
                )
            plt.plot(
                x_axis_model,
                x_axis_model**(4/3)*20,
                color = "black")
        
        plt.xlabel("Rel. indentation")
        plt.ylabel("Force ($\mu$N)")
        plt.title("Smoothed force")
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(0.001,100)
        plt.xlim(0.001,2)
        plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False)
        plt.tight_layout()
        
        outpath = self.batch_folders["Batch_Plots"] / "Smoothed_Force_Overlay_log.svg"
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        
        plt.figure(figsize=self.figsize)
        for i, (name, analyzer) in enumerate(self.analyzers.items()):
            displacement = analyzer.peak_data["Displacement [um]"].values
            peak_force = analyzer.peak_data["Force A [uN]"].values
            
            plt.plot(
                displacement,
                peak_force,
                color=self.color_overlays[i],
                )
        
        plt.xlabel("Displacement ($\mu$m)")
        plt.ylabel("Force ($\mu$N)")
        plt.title("Force")
        plt.ylim(-4,34)
        plt.xlim(0,5)
        plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False)
        plt.tight_layout()
        
        outpath = self.batch_folders["Batch_Plots"] / "Batch_Force_vs_Displacement_Peak Data_Overlay.svg"
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        
        #%%4th plot only small indentation range
        plt.figure(figsize=self.figsize)
        for i, (name, analyzer) in enumerate(self.analyzers.items()):
            
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
            smoothed_peak_force = data["Force A [uN]"].values
            
            rel_indent = displacement/(analyzer.cell_diameter/2) #(Lacorre2024)
            rel_indent_below_10_perc = rel_indent[rel_indent<= 0.1]
            force_below_10_perc = smoothed_peak_force[rel_indent<=0.1]
            x_axis_model = np.logspace(-3,1,100)
            
            plt.plot(
                rel_indent_below_10_perc,
                force_below_10_perc,
                color=self.color_overlays[i],
                label=name,
                linestyle = "",
                marker = "o"
                )
            plt.plot(
                x_axis_model,
                x_axis_model**(3/2)*20,
                color = "black")
        
        plt.xlabel("Rel. indentation")
        plt.ylabel("Force ($\mu$N)")
        plt.title("Smoothed force")
        plt.xscale('log')
        plt.yscale('log')
        #plt.ylim(0.013,14)
        #plt.xlim(0.01,0.1)
        plt.tight_layout()
        
        outpath = self.batch_folders["Batch_Plots"] / "Smoothed_Force_Overlay_log_1_perc_indent.svg"
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        
        #%% 5th plot peak data overlay
        plt.figure(figsize=self.figsize)
        for i, (name, analyzer) in enumerate(self.analyzers.items()):
            displacement = analyzer.peak_data["Displacement [um]"].values
            peak_force = analyzer.peak_data["Force A [uN]"].values
            
            plt.plot(
                displacement,
                peak_force,
                color=self.color_overlays[i],
                )
        
        plt.xlabel("Displacement ($\mu$m)")
        plt.ylabel("Force ($\mu$N)")
        plt.title("Force")
        plt.ylim(-4,34)
        plt.xlim(0,5)
        plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False)
        plt.tight_layout()
        
        outpath = self.batch_folders["Batch_Plots"] / "Batch_Force_vs_Displacement_Peak Data_Overlay.svg"
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        #%%6th plot Stiffness low indentation
        for i, (name, analyzer) in enumerate(self.analyzers.items()):
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
            smoothed_force_per_displacement = data["Force A [uN]"].values
            
            if analyzer.cell_diameter == 0:
                print(f"Analyzer {name} has cell diameter 0 import valid cell diameter\n")
                break
            
            rel_indent = displacement/(analyzer.cell_diameter/2)
            rel_indent_below_10_perc = rel_indent[rel_indent<= 0.1]
            force_per_displacement_below_10_perc = smoothed_force_per_displacement[rel_indent<= 0.1]
            
            plt.plot(
                rel_indent_below_10_perc,
                force_per_displacement_below_10_perc,
                color=colors[i],
                label=name,
                linestyle = "",
                marker = "o"
                )
        
        plt.xlabel("Rel. indentation")
        plt.ylabel("Stiffness ($\mu$N/$\mu$m)")
        plt.title("Smoothed slope")
        plt.ylim(-2,10)
        plt.xlim(0,0.1)
        #plt.legend(
        #loc="center left",
        #bbox_to_anchor=(1.02, 0.5),
        #frameon=False)
        plt.tight_layout()
        
        outpath = self.batch_folders["Batch_Plots"] / "Smoothed_Slope_Overlay.svg"
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
    
    #%% Preprocessing
    def smooth_force_time_all(self,med_kernel = 17, sg_window= 40, sg_order= 2,plot_true = False):
        """
        Smooth and despike all force_time curves
        """
        for name, analyzer in self.analyzers.items():
            analyzer.smooth_force_time(med_kernel = med_kernel, sg_window = sg_window, sg_order = sg_order,plot_true = plot_true)
    
    
    def rolling_slope_all(self,window = 5, plot_true = False):
        """"
        Calculate rolling slope for all smoothed force_time curves
        """
        for name, analyzer in self.analyzers.items():
            analyzer.rolling_slope(window = window,plot_true = plot_true)
            
    def detect_contact_slope_all(self, start_noise = 10, r_noise = 0.5,min_noise = 70,threshold_sigma=5, plot_true = False):
        """
        Detect Peak for all samples
        """
        
        for name, analyzer in self.analyzers.items():
            analyzer.detect_contact_slope(start_noise = start_noise, r_noise = r_noise,min_noise = min_noise,threshold_sigma=threshold_sigma,plot_true = plot_true)

    def subtract_baseline_all(self, margin = 10, plot_true = False):
        for name,analyzer in self.analyzers.items():
            analyzer.subtract_baseline(margin = margin, plot_true = plot_true)
            
    def bin_force_displacement_all(self, data=None, bin_width=0.05, 
                              method="median", min_points=3, 
                              compute_spread=True, plot_true=False):
        for name,analyzer in self.analyzers.items():
            analyzer.bin_force_displacement(data=data, bin_width=bin_width, 
                                      method=method, min_points=min_points, 
                                      compute_spread=compute_spread, plot_true=plot_true)
            
    def gaussian_force_displacement_all(
            self,
            data=None,
            sigma=1.0,
            min_points=3,
            compute_spread=True,
            plot_true=False
            ):
        for name,analyzer in self.analyzers.items():
            analyzer.gaussian_force_displacement(
                    data=data,
                    sigma=sigma,
                    min_points=min_points,
                    compute_spread=compute_spread,
                    plot_true=plot_true
                    )
            
            
    def cutoff_all2(self, min_fraction=0.7,drop_limit = 5,factor_stddev = 1, plot_true=False):
        for name,analyzer in self.analyzers.items():
            analyzer.cutoff2(min_fraction=min_fraction,drop_limit = drop_limit, factor_stddev = factor_stddev, plot_true=plot_true)
            
    def check_normality_slopes_baseline_all(self):
        for name,analyzer in self.analyzers.items():
            analyzer.check_normality_slopes_baseline()
            
    def shift_data_all(self):
        for name,analyzer in self.analyzers.items():
            analyzer.shift_data()

#%% Cell properties
    def average_cell_diameter(self):
        diameters = []
        
        for name,analyzer in self.analyzers.items():
            diameters.append(analyzer.cell_diameter)
        
        diameters_np = np.array(diameters)
        av_diameter = np.average(diameters_np)
        sd_diameter = np.std(diameters_np,ddof = 1)
        
        self._plot_boxplot(diameters_np, "Cell diameters")
        
        self.av_cell_diameter= av_diameter
        self.sd_cell_diameter= sd_diameter
        
        print(f"The average cell diameter is {av_diameter} $\mu$m with std.dev {sd_diameter}")
        
        return(av_diameter,sd_diameter)
        

    #%% Analysis on peaks
            
            
    def calc_bursting_force_energy_all(self,label_add = "", data_str = None, plot_true = False):
        for name,analyzer in self.analyzers.items():
            analyzer.calc_bursting_force_energy(label_add2 = label_add, data_str = data_str, plot_true = plot_true)
            
    #%% Energy and Force and Displacement to disrupt
            
    def average_bursting_forces_displacements_energy(self,gauss_sigma):
        bursting_forces = []
        bursting_energies = []
        bursting_displacements = []
        energy_per_dw = []
        
        bursting_forces_per_diameter= []
        bursting_energies_per_diameter= []
        bursting_displacements_per_diameter= []
        
        for name,analyzer in self.analyzers.items():
            bursting_forces.append(analyzer.bursting_force)
            bursting_energies.append(analyzer.bursting_energy)
            bursting_displacements.append(analyzer.bursting_displacement)
            energy_per_dw.append(analyzer.energy_dw)
            
            bursting_forces_per_diameter.append(analyzer.bursting_force_per_diameter)
            bursting_energies_per_diameter.append(analyzer.bursting_energy_per_diameter)
            bursting_displacements_per_diameter.append(analyzer.bursting_displacement_per_diameter)
            
        bursting_forces_np = np.array(bursting_forces)
        bursting_energies_np = np.array(bursting_energies)
        bursting_displacements_np = np.array(bursting_displacements)
        energy_per_dw_np = np.array(energy_per_dw)
        bursting_forces_np_per_diameter = np.array(bursting_forces_per_diameter)
        bursting_energies_np_per_diameter = np.array(bursting_energies_per_diameter)
        bursting_displacements_np_per_diameter = np.array(bursting_displacements_per_diameter)
        
        av_bursting_forces = np.average(bursting_forces_np)
        sd_bursting_forces = np.std(bursting_forces_np,ddof = 1)
        av_bursting_energies = np.average(bursting_energies_np)
        sd_bursting_energies = np.std(bursting_energies_np,ddof = 1)
        av_energy_per_dw = np.average(energy_per_dw_np)
        sd_energy_per_dw = np.std(energy_per_dw_np,ddof = 1)
        av_bursting_displacements = np.average(bursting_displacements_np)
        sd_bursting_displacements = np.std(bursting_displacements_np,ddof = 1)
        
        av_bursting_forces_per_diameter = np.average(bursting_forces_np_per_diameter)
        sd_bursting_forces_per_diameter = np.std(bursting_forces_np_per_diameter,ddof = 1)
        av_bursting_energies_per_diameter = np.average(bursting_energies_np_per_diameter)
        sd_bursting_energies_per_diameter = np.std(bursting_energies_np_per_diameter,ddof = 1)
        av_bursting_displacements_per_diameter = np.average(bursting_displacements_np_per_diameter)
        sd_bursting_displacements_per_diameter = np.std(bursting_displacements_np_per_diameter,ddof = 1)
        

        print(f"The average bursting force is {av_bursting_forces} muN with std.dev {sd_bursting_forces} muN\n The average bursting energy is {av_bursting_energies:.2f} pJ with std.dev {sd_bursting_energies:.2f} pJ\n The average bursting displacement is {av_bursting_displacements:.2f} $\mu$m with std.dev {sd_bursting_displacements:.2f}\n The average bursting force per diameter is {av_bursting_forces_per_diameter:.2f} $\mu$N/$\mu$m with std.dev {sd_bursting_forces_per_diameter:.2f} muN/mum\n The average bursting energy per diameter is {av_bursting_energies_per_diameter:.2f} pJ/$\mu$m with std.dev {sd_bursting_energies_per_diameter:.2f} pJ/$\mu$m\n The average bursting displacement per diameter is {av_bursting_displacements_per_diameter:.2f} mum/mum with std.dev {sd_bursting_displacements_per_diameter}\n")
        self.av_bursting_force = av_bursting_forces
        self.sd_bursting_force = sd_bursting_forces
        
        self.av_bursting_energy = av_bursting_energies
        self.sd_bursting_energy = sd_bursting_energies
        
        self.av_bursting_displacement = av_bursting_displacements
        self.sd_bursting_displacement = sd_bursting_displacements
        
        self.av_energy_per_dw = av_energy_per_dw
        self.sd_energy_per_dw = sd_energy_per_dw
        
        self.av_bursting_force_per_diameter = av_bursting_forces_per_diameter
        self.sd_bursting_force_per_diameter = sd_bursting_forces_per_diameter
        
        self.av_bursting_energy_per_diameter = av_bursting_energies_per_diameter
        self.sd_bursting_energy_per_diameter = sd_bursting_energies_per_diameter
        
        self.av_bursting_displacement_per_diameter = av_bursting_displacements_per_diameter
        self.sd_bursting_displacement_per_diameter = sd_bursting_displacements_per_diameter
        
        self._plot_boxplot(bursting_forces_np, f"Bursting Force Gaussian Sigma {gauss_sigma:.2f}")
        self._plot_boxplot(bursting_energies_np, f"Bursting Energy Gaussian Sigma {gauss_sigma:.2f}")
        self._plot_boxplot(bursting_displacements_np, f"Bursting Displacement Gaussian Sigma {gauss_sigma:.2f}")
        
        self._plot_boxplot(bursting_forces_np_per_diameter, f"Bursting Force per diameter Gaussian Sigma {gauss_sigma:.2f}")
        self._plot_boxplot(bursting_energies_np_per_diameter, f"Bursting Energy per diameter Gaussian Sigma {gauss_sigma:.2f}")
        self._plot_boxplot(bursting_displacements_np_per_diameter, f"Bursting Displacement per diameter Gaussian Sigma {gauss_sigma:.2f}")
        
        # ===============================
        # EXPORT CSV FILES
        # ===============================
        
        batch_results_path = self.batch_folders["Batch_Results"]
        
        # ---------
        # Individual values per cell
        # ---------
        rows = []
        
        for name, analyzer in self.analyzers.items():
            rows.append({
                "cell_id": name,
                "bursting_force_uN": analyzer.bursting_force,
                "bursting_energy_pJ": analyzer.bursting_energy,
                "bursting_displacement_um": analyzer.bursting_displacement,
                "energy_per_dw_J_per_kg": analyzer.energy_dw,
                "bursting_force_per_diameter_uN_per_um": analyzer.bursting_force_per_diameter,
                "bursting_energy_per_diameter_pJ_per_um": analyzer.bursting_energy_per_diameter,
                "bursting_displacement_per_diameter_um_per_um": analyzer.bursting_displacement_per_diameter
            })
        
        df_individual = pd.DataFrame(rows)
        df_individual.to_csv(
            batch_results_path / f"bursting_individual_values_Gaussian_Sigma_{gauss_sigma:.2f}_.csv",
            index=False
        )
        
        # ---------
        # Population averages
        # ---------
        df_avg = pd.DataFrame([{
            "N_cells": len(bursting_forces_np),
        
            "avg_bursting_force_uN": av_bursting_forces,
            "sd_bursting_force_uN": sd_bursting_forces,
        
            "avg_bursting_energy_pJ": av_bursting_energies,
            "sd_bursting_energy_pJ": sd_bursting_energies,
        
            "avg_bursting_displacement_um": av_bursting_displacements,
            "sd_bursting_displacement_um": sd_bursting_displacements,
        
            "avg_energy_per_dw_J_per_kg": av_energy_per_dw,
            "sd_energy_per_dw_J_per_kg": sd_energy_per_dw,
        
            "avg_bursting_force_per_diameter_uN_per_um": av_bursting_forces_per_diameter,
            "sd_bursting_force_per_diameter_uN_per_um": sd_bursting_forces_per_diameter,
        
            "avg_bursting_energy_per_diameter_pJ_per_um": av_bursting_energies_per_diameter,
            "sd_bursting_energy_per_diameter_pJ_per_um": sd_bursting_energies_per_diameter,
        
            "avg_bursting_displacement_per_diameter_um_per_um": av_bursting_displacements_per_diameter,
            "sd_bursting_displacement_per_diameter_um_per_um": sd_bursting_displacements_per_diameter
        }])
        
        df_avg.to_csv(
            batch_results_path / f"bursting_population_averages_Gauss_Sigma{gauss_sigma:.2f}_.csv",
            index=False
        )
        
        if self.verbose:
            print("Bursting metrics exported to CSV in Batch_Results.")
        
        return(av_bursting_forces,
               sd_bursting_forces, 
               av_bursting_energies,
               sd_bursting_energies,
               av_bursting_displacements, 
               sd_bursting_displacements,
               av_bursting_forces_per_diameter,
               sd_bursting_forces_per_diameter,
               av_bursting_energies_per_diameter,
               sd_bursting_energies_per_diameter,
               av_bursting_displacements_per_diameter,
               sd_bursting_displacements_per_diameter
               )
            
    #%% Analysis on All data
    
    def _plot_QQ(self,char_param,char_param_name):# Check normality of all medians with QQ plot, input is np array with char param and name of the param, string
        plt.figure(figsize =self.figsize) # Define e.g. figure size
        stats.probplot(char_param, dist="norm", plot=plt)
        plt.title(char_param_name)
        plt.gca().get_lines()[0].set_color(self.color(0.0))  # Change scatter points
        plt.gca().get_lines()[1].set_color(self.color(0.5))  # Change diagonal line
        outpath = self.batch_folders["Batch_Plots"] / f"QQ_plot_{char_param_name}.svg"
        plt.savefig(outpath, dpi = 300)
        plt.show()
        plt.close()
        
    def _plot_boxplot(self, char_param, char_param_name):
        plt.figure(figsize=self.figsize)
        
        
        # Underlay datapoints with jitter
        x = np.ones(len(char_param))
        jitter = 0.04 * (np.random.rand(len(char_param)) - 0.5)
        plt.plot(x + jitter, char_param, '.', color = self.color(0.5))
        #Boxplot
        plt.boxplot(char_param, vert=True)
        
        plt.ylabel(char_param_name)
        plt.title(f"{char_param_name} distribution")
        outpath = self.batch_folders["Batch_Plots"] / f"Boxplot_{char_param_name}.svg"
        plt.savefig(outpath, dpi=300)
        plt.show()
        plt.close()
        
    def _export_stats(self, char_param, char_param_name):
        stats_dict = {
            "mean": np.mean(char_param),
            "median": np.median(char_param),
            "std": np.std(char_param, ddof=1),
            "min": np.min(char_param),
            "Q10": np.quantile(char_param,0.1),
            "Q90": np.quantile(char_param,0.9),
            "max": np.max(char_param),
            "n": len(char_param),
    }
        
        print(f"\nStatistics for {char_param_name}:")
        for k, v in stats_dict.items():
            if k == "n":
                print(f"  {k:>6s}: {v:d}")
            else:
                print(f"  {k:>6s}: {v:.4g}")
        
        
        outpath = self.batch_folders["Batch_Results"] / f"Statistics_{char_param_name}.csv"
        pd.DataFrame(stats_dict, index=[0]).to_csv(outpath, index=False)
        
        
    def plot_force_mean_std(self, datatype="gauss", xmode="displacement", n_grid=500,min_analyzers = 21,x_low = -0.025, x_high = 4.25, y_low = -0.2, y_high = 46):
        """
        Plots mean ± std of force curves across analyzers with interpolation.
        Handles different x-ranges and lengths per curve.
        min_analyzers states how many analyzers need to be available at for mean/sd curve
        
        """
    
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.interpolate import interp1d
        
        
    
        if xmode not in X_AXIS_CONFIG:
            raise ValueError(f"xmode must be one of {list(X_AXIS_CONFIG.keys())}")
    
        forces_interp = []
        x_min, x_max = np.inf, -np.inf
    
        x_key = X_AXIS_CONFIG[xmode]["col"]
    
        # -------------------------------------------------
        # 1. First pass: determine global x-range
        # -------------------------------------------------
        for analyzer in self.analyzers.values():
    
            if datatype == "gauss":
                data = analyzer.pp_gauss
            elif datatype == "binned":
                data = analyzer.pp_binned
            elif datatype == "peak":
                data = analyzer.peak_data
            else:
                raise ValueError("datatype must be: gauss, binned, peak")
    
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
        for name, analyzer in self.analyzers.items():
            if datatype == "gauss":
                data = analyzer.pp_gauss
            elif datatype == "binned":
                data = analyzer.pp_binned
            elif datatype == "peak":
                data = analyzer.peak_data
            else:
                raise ValueError("datatype must be: gauss, binned, peak")
                
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
        
        # initialize outputs
        median_force = np.full(forces_interp.shape[1], np.nan)
        iqr_lower = np.full(forces_interp.shape[1], np.nan)
        iqr_upper = np.full(forces_interp.shape[1], np.nan)
        
        #compute only where enough data exists
        median_force[valid_mask] = np.nanmedian(
            forces_interp[:, valid_mask],
            axis=0
        )
        
        median_force = median_force[valid_mask]
        
        q25 = np.nanpercentile(forces_interp[:, valid_mask], 25, axis=0)
        q75 = np.nanpercentile(forces_interp[:, valid_mask], 75, axis=0)
        
        iqr_lower = q25
        iqr_upper = q75
    
        x_common = x_common[valid_mask]
    
        # -------------------------------------------------
        # 4. Plot
        # -------------------------------------------------
        plt.figure(figsize =self.figsize)
    
        plt.plot(
            x_common,
            median_force,
            color = self.color(0.5),
            label="Median"
        )

        plt.fill_between(
            x_common,
            iqr_lower,
            iqr_upper,
            color = self.color(0.5),
            alpha=0.3,
            label="IQR (25-75%)"
        )
    
        plt.xlabel(X_AXIS_CONFIG[xmode]["label"])
        plt.ylabel("Force ($\mu$N)")
        plt.legend()
        plt.xlim(x_low,x_high)
        plt.ylim(y_low,y_high)
        plt.tight_layout()
        outpath = self.batch_folders["Batch_Plots"] / f"median_IQR_{datatype}_{xmode}.svg"
        plt.savefig(outpath, dpi=300)
        plt.show()
        plt.close()
    
    