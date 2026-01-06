#!/usr/bin/env python
# coding: utf-8

import argparse
import glob
import os
import warnings
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import datetime
import pandas as pd

# Constants
G = 4.30091e-6  # Gravitational constant in kpc/M_sun * (km/s)^2
H0 = 70.0  # Hubble constant in km/s/Mpc

# --- Model Definitions ---

def V_lambda_cdm(r, V_char, r_s):
    """
    NFW halo velocity profile using characteristic velocity and scale radius.
    
    V^2(r) = V_char^2 * (r_s/r) * [ln(1+r/r_s) - (r/r_s)/(1+r/r_s)]
    
    Args:
        r (array_like): Radius in kpc.
        V_char (float): Characteristic velocity in km/s.
        r_s (float): Scale radius in kpc.
        
    Returns:
        array_like: Velocity in km/s.
    """
    if V_char <= 0 or r_s <= 0: return np.inf
    
    x = r / r_s
    
    # NFW velocity profile
    term = np.log(1 + x) - x / (1 + x)
    v_sq = V_char**2 * (1/x) * term
    
    return np.sqrt(np.maximum(0, v_sq))

def V_svt(r, V_infty, r_c, M_bary):
    """
    SVT velocity profile from Logarithmic Superfluid Vacuum Theory.
    
    Derived from the Log-NLSE: iℏ∂ψ/∂t = -ℏ²/2m ∇²ψ - b·ln(|ψ|²/ρ₀)·ψ
    
    The logarithmic nonlinearity produces:
    - Isothermal equation of state: P = bρ
    - Density-independent sound speed: c_s = √(b/m)
    - Logarithmic spatial profile for rotation curves
    
    Formula:
        V(r) = V_∞ · √[ln(1 + r/r_c)] / √[ln(1 + R_norm/r_c)] · [ln(1 + M/M_c)]^0.25
    
    Args:
        r (array_like): Radius in kpc
        V_infty (float): Asymptotic velocity scale (km/s)
        r_c (float): Core radius / effective healing length (kpc)
        M_bary (float): Total baryonic mass (M_sun)
        
    Returns:
        array_like: Vacuum contribution to rotation velocity (km/s)
    """
    M_crit = 1.0e6   # Critical mass scale (M_sun)
    M_crit = 1.0e6   # Critical mass scale (M_sun)
    
    # Convert r to array if scalar
    r = np.atleast_1d(r)
    
    # Parameter validation - return array of inf for invalid params
    if V_infty <= 0 or r_c <= 0:
        return np.full_like(r, np.inf, dtype=float)
    
    # Protect against numerical issues
    r_safe = np.maximum(r, 1e-10)  # Avoid r=0
    
    # Logarithmic spatial profile (from Log-NLSE pressure term)
    # The isothermal P = bρ gives ∇P/ρ = b∇(ln ρ), leading to sqrt(ln) profile
    arg = 1.0 + r_safe / r_c
    spatial = np.sqrt(np.maximum(np.log(arg), 1e-10))
    
    # Mass-dependent factor for BTFR scaling
    # γ = 0.25 gives v⁴ ∝ M (Baryonic Tully-Fisher Relation)
    if M_bary <= 0:
        return np.full_like(r, np.inf, dtype=float)
    
    mass_arg = 1.0 + M_bary / M_crit
    mass_factor = np.power(np.log(mass_arg), 0.25)
    
    return V_infty * spatial * mass_factor

# --- Data Loading ---

def calculate_baryonic_mass(Radius, V_gas, V_disk, V_bulge):
    """
    Estimates the total baryonic mass from the rotation curve components.
    
    Uses a point-mass approximation at the last measured radius:
    M_bary approx (V_gas^2 + V_disk^2 + V_bulge^2) * R_last / G
    
    Note: This is an approximation sufficient for establishing the "Alpha Correlation" 
    trend but treats the extended mass distribution as a point mass at the edge.
    """
    if len(Radius) == 0:
        return None
    r_last = Radius[-1]
    v_bary_sq_last = V_gas[-1]**2 + V_disk[-1]**2 + V_bulge[-1]**2
    mass = v_bary_sq_last * r_last / G
    return mass

def load_sparc_galaxy(filepath):
    """
    Loads a single galaxy data file from the SPARC database.
    Checks for NaN or Inf values and skips if found.
    """
    try:
        data = np.loadtxt(filepath)
        
        # Check for NaN or Inf
        if not np.all(np.isfinite(data)):
            # print(f"Warning: NaN or Inf found in {filepath}")
            return None, None, None, None, None, None

        galaxy_name = os.path.basename(filepath).replace('_rotmod.dat', '')
        
        Radius = data[:, 0]
        V_obs = data[:, 1]
        V_err = data[:, 2]
        V_gas = data[:, 3]
        V_disk = data[:, 4]
        V_bulge = data[:, 5]

        V_bary = np.sqrt(np.maximum(0, V_gas**2 + V_disk**2 + V_bulge**2))
        mass = calculate_baryonic_mass(Radius, V_gas, V_disk, V_bulge)

        return galaxy_name, Radius, V_obs, V_err, V_bary, mass
    except (IOError, IndexError, ValueError):
        return None, None, None, None, None, None

# --- Plotting ---

def plot_galaxy_fit(name, r, v_obs, v_err, v_bary, popt_lambda_cdm, chi2_nu_lambda_cdm, popt_svt, chi2_nu_svt, mass, output_dir):
    """
    Generates and saves a detailed plot for a single galaxy fit.
    """
    plt.figure(figsize=(12, 8))
    
    plt.errorbar(r, v_obs, yerr=v_err, fmt='k.', label='SPARC Data', capsize=3, alpha=0.7, zorder=1)

    # Use a list to build the text string for better formatting control
    text_lines = [f"Galaxy: {name}"]

    # Lambda CDM Model
    if popt_lambda_cdm is not None and np.isfinite(chi2_nu_lambda_cdm):
        v_halo_lambda_cdm = V_lambda_cdm(r, *popt_lambda_cdm)
        v_total_lambda_cdm = np.sqrt(v_bary**2 + v_halo_lambda_cdm**2)
        plt.plot(r, v_total_lambda_cdm, color='blue', lw=2, label='Total Lambda CDM Fit', zorder=3)
        text_lines.append(f"Lambda CDM: $\chi_\\nu^2$ = {chi2_nu_lambda_cdm:.2f}")
        text_lines.append(f"  $V_{{char}}={popt_lambda_cdm[0]:.1f}$ km/s")
        text_lines.append(f"  $r_s={popt_lambda_cdm[1]:.1f}$ kpc")
    else:
        text_lines.append("Lambda CDM Fit: Failed")

    text_lines.append("") # Spacer

    # SVT Model
    if popt_svt is not None and np.isfinite(chi2_nu_svt):
        v_halo_svt = V_svt(r, *popt_svt, M_bary=mass)
        v_total_svt = np.sqrt(v_bary**2 + v_halo_svt**2)
        plt.plot(r, v_total_svt, color='red', lw=2, label='Total SVT Fit', zorder=4)
        text_lines.append(f"SVT: $\chi_\\nu^2$ = {chi2_nu_svt:.2f}")
        text_lines.append(f"  $V_{{\infty}}={popt_svt[0]:.1f}$ km/s")
        text_lines.append(f"  $r_{{c}}={popt_svt[1]:.2f}$ kpc")
    else:
        text_lines.append("SVT Fit: Failed")

    text_info = "\n".join(text_lines)

    plt.title(f'Rotation Curve Fit for {name}', fontsize=16, weight='bold')
    plt.xlabel('Radius (kpc)', fontsize=12)
    plt.ylabel('Velocity (km/s)', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    # Improve text box placement and appearance
    plt.figtext(0.15, 0.85, text_info, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'), fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = os.path.join(output_dir, f"{name}_fit.png")
    plt.savefig(output_path)
    plt.close()

# --- Fitting and Analysis ---

def calculate_chi2_nu(V_obs, V_model, V_err, k):
    """Calculates the reduced chi-squared."""
    N = len(V_obs)
    if N - k <= 0: return np.inf
    
    # Handle zero errors to avoid division by zero
    epsilon = 1e-9
    safe_V_err = np.where(V_err <= 0, epsilon, V_err)
    
    chi2 = np.sum(((V_obs - V_model) / safe_V_err)**2)
    return chi2 / (N - k)

def fit_model_to_galaxy(Radius, V_obs, V_err, V_bary, model_func, p0, bounds, mass=None):
    """
    Fits a given halo model to a galaxy's rotation curve.
    Handles the special case for the TVVT model which requires baryonic mass.
    """
    if model_func == V_svt:
        if mass is None or mass <= 0:
            # This case should be caught earlier, but as a safeguard:
            return None, np.inf
        # Create a model wrapper that fixes the mass parameter for the fit function
        model_for_fit = lambda r, *params: model_func(r, *params, M_bary=mass)
    else:
        model_for_fit = model_func

    def fit_func(r, *params):
        return np.sqrt(V_bary**2 + model_for_fit(r, *params)**2)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(fit_func, Radius, V_obs, p0=p0, sigma=V_err, bounds=bounds, maxfev=8000, method='trf')
            
        V_model = fit_func(Radius, *popt)
        chi2_nu = calculate_chi2_nu(V_obs, V_model, V_err, k=len(p0))
        return popt, chi2_nu
    except (RuntimeError, ValueError):
        return None, np.inf



def main():
    """Main analysis script."""
    parser = argparse.ArgumentParser(description="Compare Lambda-CDM and SVT models using SPARC galaxy data.")
    parser.add_argument("--data_dir", type=str, default="sparc_data/sparc_database/", help="Path to the SPARC data directory.")
    parser.add_argument("--output_dir", type=str, default="galaxy_fits", help="Directory to save output plots.")
    args = parser.parse_args()

    output_dir = args.output_dir
    data_dir = args.data_dir

    os.makedirs(output_dir, exist_ok=True)
    print(f"Individual galaxy plots will be saved to '{output_dir}/'")

    files = sorted(glob.glob(os.path.join(data_dir, "*_rotmod.dat")))

    if not files:
        print(f"No data files found in {data_dir}. Please check the path.")
        return

    results_lambda_cdm = []
    results_svt = []
    
    print(f"Starting detailed analysis on {len(files)} galaxies...")

    # Prepare list for CSV
    csv_data = []

    for i, filepath in enumerate(files):
        name, r, v_obs, v_err, v_bary, mass = load_sparc_galaxy(filepath)
        
        if name is None or len(r) < 3:
            print(f"  Skipping {os.path.basename(filepath)} (bad data)")
            continue
            
        # Skip galaxies with invalid mass, as per the new model requirements
        if mass is None or mass <= 0:
            print(f"  Skipping {name} due to invalid baryonic mass ({mass}).")
            # Still generate a plot showing the data and failed fit status
            plot_galaxy_fit(name, r, v_obs, v_err, v_bary, None, np.inf, None, np.inf, mass, output_dir)
            continue

        popt_lambda_cdm, chi2_nu_lambda_cdm = fit_model_to_galaxy(r, v_obs, v_err, v_bary, V_lambda_cdm, p0=[100, 10], bounds=([1, 0.1], [500, 50]))

        # Fit SVT with the new model, passing mass and using new bounds
        popt_svt, chi2_nu_svt = fit_model_to_galaxy(
            r, v_obs, v_err, v_bary, 
            V_svt, 
            p0=[200, 10.0],  # Start guess for V_infty, r_c
            bounds=([10, 0.1], [500, 100]), 
            mass=mass
        )

        plot_galaxy_fit(name, r, v_obs, v_err, v_bary, popt_lambda_cdm, chi2_nu_lambda_cdm, popt_svt, chi2_nu_svt, mass, output_dir)

        if popt_lambda_cdm is not None:
            results_lambda_cdm.append({"galaxy": name, "params": popt_lambda_cdm, "chi2_nu": chi2_nu_lambda_cdm})
            
        if popt_svt is not None:
            # Store the mass along with the results for the final plot
            results_svt.append({"galaxy": name, "params": popt_svt, "chi2_nu": chi2_nu_svt, "mass": mass})
        
        # Collect data for CSV
        csv_data.append({
            "Galaxy": name,
            "Chi2_Lambda_CDM": chi2_nu_lambda_cdm if popt_lambda_cdm is not None else np.nan,
            "Chi2_SVT": chi2_nu_svt if popt_svt is not None else np.nan,
            "V_infty": popt_svt[0] if popt_svt is not None else np.nan,
            "r_c": popt_svt[1] if popt_svt is not None else np.nan
        })

        if (i+1) % 10 == 0:
            print(f"  ...processed {i+1}/{len(files)} galaxies (plots generated).")

    print("Analysis complete. Generating final results...")
    
    # --- Save CSV ---
    df = pd.DataFrame(csv_data)
    df.to_csv("galaxy_chi2_comparison.csv", index=False)
    print("  -> 'galaxy_chi2_comparison.csv' generated.")

    chi2_lambda_cdm_values = np.array([res['chi2_nu'] for res in results_lambda_cdm if np.isfinite(res['chi2_nu'])])
    chi2_svt_values = np.array([res['chi2_nu'] for res in results_svt if np.isfinite(res['chi2_nu'])])

    # Filter for better visualization in histogram (remove extreme outliers)
    limit = 20 # Focus on the "good fit" region
    chi2_lambda_cdm_filtered = chi2_lambda_cdm_values[chi2_lambda_cdm_values < limit]
    chi2_svt_filtered = chi2_svt_values[chi2_svt_values < limit]

    plt.figure(figsize=(12, 7))
    plt.hist(chi2_lambda_cdm_filtered, bins=20, alpha=0.7, label=f'Lambda CDM (N={len(chi2_lambda_cdm_filtered)}/{len(chi2_lambda_cdm_values)})', color='blue', density=True)
    plt.hist(chi2_svt_filtered, bins=20, alpha=0.7, label=f'SVT (N={len(chi2_svt_filtered)}/{len(chi2_svt_values)})', color='red', density=True)
    plt.xlabel('Reduced Chi-Squared ($\chi_\nu^2$)', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    plt.title(f'Final Showdown: Lambda CDM vs. SVT (Zoomed: $\chi_\\nu^2$ < {limit})', fontsize=16, weight='bold')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("final_showdown.png")
    print("  -> 'final_showdown.png' generated.")

    # --- Universality Check: V_infty vs Mass ---
    print("\nGenerating Universality Check Plot...")
    
    valid_results = [res for res in results_svt if 'mass' in res and res['mass'] is not None and res['mass'] > 0]
    masses = [res['mass'] for res in valid_results]
    V_infty_values = [res['params'][0] for res in valid_results]
    r_c_values = [res['params'][1] for res in valid_results]
            
    if len(masses) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # V_infty vs Mass
        axes[0].scatter(masses, V_infty_values, alpha=0.6, c='darkred', edgecolors='k')
        axes[0].set_xscale('log')
        axes[0].set_xlabel('Baryonic Mass ($M_{\odot}$)', fontsize=12)
        axes[0].set_ylabel('$V_{\infty}$ (km/s)', fontsize=12)
        axes[0].set_title('SVT: Asymptotic Velocity vs Mass', fontsize=12, weight='bold')
        axes[0].grid(True, which="both", ls="--", alpha=0.3)
        if len(V_infty_values) > 0:
            mean_V = np.mean(V_infty_values)
            axes[0].axhline(mean_V, color='r', linestyle='--', lw=2, 
                           label=f'Mean $V_\\infty$ = {mean_V:.1f} km/s')
            axes[0].legend()
        
        # r_c vs Mass
        axes[1].scatter(masses, r_c_values, alpha=0.6, c='darkblue', edgecolors='k')
        axes[1].set_xscale('log')
        axes[1].set_xlabel('Baryonic Mass ($M_{\odot}$)', fontsize=12)
        axes[1].set_ylabel('$r_c$ (kpc)', fontsize=12)
        axes[1].set_title('SVT: Core Radius vs Mass', fontsize=12, weight='bold')
        axes[1].grid(True, which="both", ls="--", alpha=0.3)
        if len(r_c_values) > 0:
            mean_rc = np.mean(r_c_values)
            axes[1].axhline(mean_rc, color='b', linestyle='--', lw=2,
                           label=f'Mean $r_c$ = {mean_rc:.1f} kpc')
            axes[1].legend()

        plt.tight_layout()
        plt.savefig("svt_parameter_universality.png")
        print("  -> 'svt_parameter_universality.png' generated.")
    else:
        print("  -> Could not generate Universality Check plot (insufficient data).")

    print("\n--- STATISTICAL VERDICT ---")
    lambda_stats = {'count': 0, 'median': 0.0, 'total_attempted': len(files)}
    svt_stats = {'count': 0, 'median': 0.0, 'total_attempted': len(files)}

    if len(chi2_lambda_cdm_values) > 0:
        median_chi2_lambda_cdm = np.median(chi2_lambda_cdm_values)
        lambda_stats['count'] = len(chi2_lambda_cdm_values)
        lambda_stats['median'] = median_chi2_lambda_cdm
        print(f"Lambda CDM:")
        print(f"  - Successfully fit {len(chi2_lambda_cdm_values)} galaxies.")
        print(f"  - Median Reduced Chi-Squared ($\\chi_\\nu^2$): {median_chi2_lambda_cdm:.3f}")
    else:
        print("Lambda CDM: No successful fits.")
        
    if len(chi2_svt_values) > 0:
        median_chi2_svt = np.median(chi2_svt_values)
        svt_stats['count'] = len(chi2_svt_values)
        svt_stats['median'] = median_chi2_svt
        print(f"SVT:")
        print(f"  - Successfully fit {len(chi2_svt_values)} galaxies.")
        print(f"  - Median Reduced Chi-Squared ($\\chi_\\nu^2$): {median_chi2_svt:.3f}")
    else:
        print("SVT: No successful fits.")
    print("---")

if __name__ == '__main__':
    main()
