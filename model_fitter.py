#!/usr/bin/env python
# coding: utf-8
"""
Galaxy Rotation Curve Fitting: Logarithmic Superfluid Vacuum Theory

Implementation based on:
    Zloshchastiev, K.G. "Galaxy rotation curves in superfluid vacuum theory"
    https://www.researchgate.net/publication/366232848

The physical vacuum is modeled as a Bose-Einstein condensate governed by the Log-NLSE:
    iℏ∂ψ/∂t = -ℏ²/2m ∇²ψ + mΦψ - b·ln(|ψ|²/ρ₀)·ψ

This produces a multi-scale gravitational potential with:
    - Newtonian term (1/r)
    - Logarithmic term (from quantum vacuum)
    - Linear/Rindler term (acceleration)
    - Quadratic/de Sitter term (cosmological)
"""

import argparse
import glob
import os
import warnings
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd

# Constants
G = 4.30091e-6  # Gravitational constant in kpc/M_sun * (km/s)^2
H0 = 70.0  # Hubble constant in km/s/Mpc
L_BAR = 1.0  # Reference length scale (kpc) for logarithmic term

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


def V_log_svt(r, b0_m, chi, k1, k2, a1_tilde, a2_tilde):
    """
    Logarithmic Superfluid Vacuum Theory (Log-SVT) velocity contribution.
    
    Based on Zloshchastiev's paper, equation (18):
    
    v²_SVT = b₀/m · R · d/dR[χ·ln(R/ℓ̄) + ln((k₂R² + k₁R + 1)²)] + ã₁R - ã₂R²
    
    The derivative gives:
    v²_SVT = b₀/m · [χ + 4R(2k₂R + k₁)/(k₂R² + k₁R + 1)] + ã₁R - ã₂R²
    
    Args:
        r (array_like): Radius in kpc
        b0_m (float): Logarithmic coupling b₀/m in km²/s²
        chi (float): Power degree for inner logarithmic divergence (dimensionless)
        k1 (float): Linear polynomial coefficient in kpc⁻¹
        k2 (float): Quadratic polynomial coefficient in kpc⁻²
        a1_tilde (float): Rindler acceleration term (km/s)²/kpc
        a2_tilde (float): de Sitter cosmological term (km/s)²/kpc²
        
    Returns:
        array_like: SVT contribution to rotation velocity squared (km²/s²)
    """
    r = np.atleast_1d(r)
    
    # Parameter validation
    if b0_m < 0:
        return np.full_like(r, np.inf, dtype=float)
    
    # Protect against numerical issues
    r_safe = np.maximum(r, 1e-10)
    
    # Polynomial P(r) = k₂R² + k₁R + 1
    P = k2 * r_safe**2 + k1 * r_safe + 1.0
    P = np.maximum(P, 1e-10)  # Ensure P > 0
    
    # Derivative of polynomial: dP/dR = 2k₂R + k₁
    dP_dR = 2.0 * k2 * r_safe + k1
    
    # Logarithmic term contribution:
    # v²_ln = b₀/m · R · d/dR[χ·ln(R/ℓ̄) + 2·ln(P)]
    #       = b₀/m · R · [χ/R + 2·dP/dR / P]
    #       = b₀/m · [χ + 2R·dP/dR / P]
    #       = b₀/m · [χ + 2R(2k₂R + k₁) / (k₂R² + k₁R + 1)]
    #       = b₀/m · [χ + 4k₂R² + 2k₁R) / P]
    
    log_term = b0_m * (chi + (4.0 * k2 * r_safe**2 + 2.0 * k1 * r_safe) / P)
    
    # Linear (Rindler) term: ã₁R (units: (km/s)²/kpc × kpc = km²/s²)
    # Note: a1_tilde can be negative (deceleration)
    linear_term = a1_tilde * r_safe
    
    # Quadratic (de Sitter) term: -ã₂R² (always with minus sign in the potential)
    # Small positive a2_tilde gives outward acceleration at large R
    quadratic_term = -a2_tilde * r_safe**2
    
    # Total v²_SVT
    v_sq_svt = log_term + linear_term + quadratic_term
    
    return v_sq_svt


def V_svt_total(r, v_bary, b0_m, chi, k1, k2, a1_tilde, a2_tilde):
    """
    Total rotation velocity combining baryonic and SVT contributions.
    
    v(R) = sqrt(v_N² + v_SVT²)
    
    where v_N is the Newtonian baryonic velocity and v_SVT is from Log-SVT.
    """
    v_sq_svt = V_log_svt(r, b0_m, chi, k1, k2, a1_tilde, a2_tilde)
    v_sq_total = v_bary**2 + v_sq_svt
    
    return np.sqrt(np.maximum(0, v_sq_total))


# --- Data Loading ---

def calculate_baryonic_mass(Radius, V_gas, V_disk, V_bulge):
    """
    Estimates the total baryonic mass from the rotation curve components.
    
    Uses a point-mass approximation at the last measured radius:
    M_bary approx (V_gas² + V_disk² + V_bulge²) * R_last / G
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

def plot_galaxy_fit(name, r, v_obs, v_err, v_bary, 
                    popt_lambda_cdm, chi2_nu_lambda_cdm, 
                    popt_svt, chi2_nu_svt, mass, output_dir):
    """
    Generates and saves a detailed plot for a single galaxy fit.
    """
    plt.figure(figsize=(14, 9))
    
    # Plot data
    plt.errorbar(r, v_obs, yerr=v_err, fmt='ko', label='SPARC Data', 
                 capsize=3, alpha=0.7, markersize=5, zorder=1)
    
    # Plot baryonic contribution
    plt.plot(r, v_bary, 'g--', lw=1.5, alpha=0.6, label='Baryonic (Newtonian)', zorder=2)

    # Text info
    text_lines = [f"Galaxy: {name}"]

    # Lambda CDM Model
    if popt_lambda_cdm is not None and np.isfinite(chi2_nu_lambda_cdm):
        v_halo_lambda_cdm = V_lambda_cdm(r, *popt_lambda_cdm)
        v_total_lambda_cdm = np.sqrt(v_bary**2 + v_halo_lambda_cdm**2)
        plt.plot(r, v_total_lambda_cdm, color='blue', lw=2, 
                 label=f'NFW (ΛCDM) χ²ᵥ={chi2_nu_lambda_cdm:.2f}', zorder=3)
        text_lines.append(f"NFW: χ²ᵥ = {chi2_nu_lambda_cdm:.2f}")
        text_lines.append(f"  V_char = {popt_lambda_cdm[0]:.1f} km/s")
        text_lines.append(f"  r_s = {popt_lambda_cdm[1]:.1f} kpc")
    else:
        text_lines.append("NFW Fit: Failed")

    text_lines.append("")

    # Log-SVT Model
    if popt_svt is not None and np.isfinite(chi2_nu_svt):
        v_sq_svt = V_log_svt(r, *popt_svt)
        v_total_svt = np.sqrt(np.maximum(0, v_bary**2 + v_sq_svt))
        plt.plot(r, v_total_svt, color='red', lw=2, 
                 label=f'Log-SVT χ²ᵥ={chi2_nu_svt:.2f}', zorder=4)
        text_lines.append(f"Log-SVT: χ²ᵥ = {chi2_nu_svt:.2f}")
        text_lines.append(f"  b₀/m = {popt_svt[0]:.1f} km²/s²")
        text_lines.append(f"  χ = {popt_svt[1]:.4f}")
        text_lines.append(f"  k₁ = {popt_svt[2]:.4f} kpc⁻¹")
        text_lines.append(f"  k₂ = {popt_svt[3]:.6f} kpc⁻²")
        text_lines.append(f"  ã₁ = {popt_svt[4]:.2e} (km/s)²/kpc")
        text_lines.append(f"  ã₂ = {popt_svt[5]:.2e} (km/s)²/kpc²")
    else:
        text_lines.append("Log-SVT Fit: Failed")

    text_info = "\n".join(text_lines)

    plt.title(f'Rotation Curve Fit: {name}', fontsize=16, weight='bold')
    plt.xlabel('Radius (kpc)', fontsize=12)
    plt.ylabel('Velocity (km/s)', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    # Text box with parameters
    plt.figtext(0.15, 0.88, text_info, 
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', 
                          boxstyle='round,pad=0.5'), 
                fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_path = os.path.join(output_dir, f"{name}_fit.png")
    plt.savefig(output_path, dpi=100)
    plt.close()


# --- Fitting and Analysis ---

def calculate_chi2_nu(V_obs, V_model, V_err, k):
    """Calculates the reduced chi-squared."""
    N = len(V_obs)
    if N - k <= 0: return np.inf
    
    # Handle zero errors
    epsilon = 1e-9
    safe_V_err = np.where(V_err <= 0, epsilon, V_err)
    
    chi2 = np.sum(((V_obs - V_model) / safe_V_err)**2)
    return chi2 / (N - k)


def fit_nfw_model(Radius, V_obs, V_err, V_bary):
    """Fits NFW (Lambda CDM) model."""
    def fit_func(r, V_char, r_s):
        v_halo = V_lambda_cdm(r, V_char, r_s)
        return np.sqrt(V_bary**2 + v_halo**2)
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(fit_func, Radius, V_obs, 
                               p0=[100, 10], 
                               sigma=V_err, 
                               bounds=([1, 0.1], [500, 50]), 
                               maxfev=8000, method='trf')
        V_model = fit_func(Radius, *popt)
        chi2_nu = calculate_chi2_nu(V_obs, V_model, V_err, k=2)
        return popt, chi2_nu
    except (RuntimeError, ValueError):
        return None, np.inf


def fit_log_svt_model(Radius, V_obs, V_err, V_bary):
    """
    Fits Logarithmic Superfluid Vacuum Theory model.
    
    Parameters: [b0_m, chi, k1, k2, a1_tilde, a2_tilde]
    
    Following paper conventions:
    - b0_m: typically 10 - 3500 km²/s²
    - chi: typically 0 (except for galaxies with non-zero inner velocity)
    - k1: typically 0 - 100 kpc⁻¹
    - k2: typically 0 - 0.3 kpc⁻²
    - a1_tilde: typically -50 to +5 × 10⁻¹¹ m/s² (convert to (km/s)²/kpc)
    - a2_tilde: typically 1e-50 to 1e-30 s⁻² (convert to (km/s)²/kpc²)
    """
    def fit_func(r, b0_m, chi, k1, k2, a1_tilde, a2_tilde):
        v_sq_svt = V_log_svt(r, b0_m, chi, k1, k2, a1_tilde, a2_tilde)
        v_sq_total = V_bary**2 + v_sq_svt
        return np.sqrt(np.maximum(0, v_sq_total))
    
    # Initial guesses based on paper's Table I
    # Use average observed velocity squared as initial b0_m
    v_avg = np.mean(V_obs)
    p0 = [v_avg**2 / 10, 0.0, 0.1, 0.01, 0.0, 1e-35]
    
    # Bounds: [lower], [upper]
    # b0_m: 1 to 5000 km²/s²
    # chi: 0 to 1 (paper shows most are 0)
    # k1: 0 to 100 kpc⁻¹
    # k2: 0 to 0.5 kpc⁻²
    # a1_tilde: -100 to 100 (km/s)²/kpc
    # a2_tilde: 0 to 1e-25 (very small de Sitter term)
    bounds = (
        [1, 0, 0, 0, -100, 0],  # lower
        [5000, 1, 100, 0.5, 100, 1e-25]  # upper
    )
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(fit_func, Radius, V_obs, 
                               p0=p0, 
                               sigma=V_err, 
                               bounds=bounds, 
                               maxfev=10000, 
                               method='trf')
        V_model = fit_func(Radius, *popt)
        chi2_nu = calculate_chi2_nu(V_obs, V_model, V_err, k=6)
        return popt, chi2_nu
    except (RuntimeError, ValueError):
        return None, np.inf


def main():
    """Main analysis script."""
    parser = argparse.ArgumentParser(
        description="Compare ΛCDM (NFW) and Log-SVT models using SPARC galaxy data.")
    parser.add_argument("--data_dir", type=str, 
                        default="sparc_data/sparc_database/", 
                        help="Path to SPARC data directory.")
    parser.add_argument("--output_dir", type=str, 
                        default="galaxy_fits", 
                        help="Directory to save output plots.")
    args = parser.parse_args()

    output_dir = args.output_dir
    data_dir = args.data_dir

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: '{output_dir}/'")
    print("=" * 60)
    print("Logarithmic Superfluid Vacuum Theory - Galaxy Rotation Curves")
    print("Based on: Zloshchastiev (2022)")
    print("=" * 60)

    files = sorted(glob.glob(os.path.join(data_dir, "*_rotmod.dat")))

    if not files:
        print(f"No data files found in {data_dir}. Please check the path.")
        return

    results_nfw = []
    results_svt = []
    csv_data = []
    
    print(f"\nAnalyzing {len(files)} galaxies from SPARC database...\n")

    for i, filepath in enumerate(files):
        name, r, v_obs, v_err, v_bary, mass = load_sparc_galaxy(filepath)
        
        if name is None or len(r) < 3:
            print(f"  [{i+1:3d}] Skipping {os.path.basename(filepath)} (bad data)")
            continue
            
        if mass is None or mass <= 0:
            print(f"  [{i+1:3d}] Skipping {name} (invalid baryonic mass)")
            plot_galaxy_fit(name, r, v_obs, v_err, v_bary, 
                           None, np.inf, None, np.inf, mass, output_dir)
            continue

        # Fit NFW (Lambda CDM)
        popt_nfw, chi2_nu_nfw = fit_nfw_model(r, v_obs, v_err, v_bary)
        
        # Fit Log-SVT
        popt_svt, chi2_nu_svt = fit_log_svt_model(r, v_obs, v_err, v_bary)

        # Generate plot
        plot_galaxy_fit(name, r, v_obs, v_err, v_bary, 
                       popt_nfw, chi2_nu_nfw, 
                       popt_svt, chi2_nu_svt, mass, output_dir)

        # Store results
        if popt_nfw is not None:
            results_nfw.append({
                "galaxy": name, 
                "params": popt_nfw, 
                "chi2_nu": chi2_nu_nfw
            })
            
        if popt_svt is not None:
            results_svt.append({
                "galaxy": name, 
                "params": popt_svt, 
                "chi2_nu": chi2_nu_svt, 
                "mass": mass
            })
        
        # CSV data
        csv_data.append({
            "Galaxy": name,
            "Chi2_NFW": chi2_nu_nfw if popt_nfw is not None else np.nan,
            "Chi2_LogSVT": chi2_nu_svt if popt_svt is not None else np.nan,
            "b0_m": popt_svt[0] if popt_svt is not None else np.nan,
            "chi": popt_svt[1] if popt_svt is not None else np.nan,
            "k1": popt_svt[2] if popt_svt is not None else np.nan,
            "k2": popt_svt[3] if popt_svt is not None else np.nan,
            "a1_tilde": popt_svt[4] if popt_svt is not None else np.nan,
            "a2_tilde": popt_svt[5] if popt_svt is not None else np.nan,
            "Mass_Msun": mass
        })

        # Progress
        winner = "NFW" if chi2_nu_nfw < chi2_nu_svt else "Log-SVT"
        if (i+1) % 10 == 0 or (i+1) == len(files):
            print(f"  [{i+1:3d}/{len(files)}] {name}: NFW={chi2_nu_nfw:.2f}, Log-SVT={chi2_nu_svt:.2f} → {winner}")

    print("\n" + "=" * 60)
    print("Analysis complete. Generating summary outputs...")
    
    # Save CSV
    df = pd.DataFrame(csv_data)
    df.to_csv("galaxy_chi2_comparison.csv", index=False)
    print("  → 'galaxy_chi2_comparison.csv' saved")

    # Get chi2 values
    chi2_nfw = np.array([res['chi2_nu'] for res in results_nfw if np.isfinite(res['chi2_nu'])])
    chi2_svt = np.array([res['chi2_nu'] for res in results_svt if np.isfinite(res['chi2_nu'])])

    # Histogram comparison
    limit = 20
    chi2_nfw_filt = chi2_nfw[chi2_nfw < limit]
    chi2_svt_filt = chi2_svt[chi2_svt < limit]

    plt.figure(figsize=(12, 7))
    plt.hist(chi2_nfw_filt, bins=25, alpha=0.6, 
             label=f'NFW (ΛCDM) [N={len(chi2_nfw_filt)}/{len(chi2_nfw)}]', 
             color='blue', density=True, edgecolor='darkblue')
    plt.hist(chi2_svt_filt, bins=25, alpha=0.6, 
             label=f'Log-SVT [N={len(chi2_svt_filt)}/{len(chi2_svt)}]', 
             color='red', density=True, edgecolor='darkred')
    plt.axvline(1.0, color='green', lw=2, ls='--', label='Ideal fit (χ²ᵥ=1)')
    plt.xlabel('Reduced Chi-Squared (χ²ᵥ)', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    plt.title('Model Comparison: NFW (ΛCDM) vs Log-SVT', fontsize=16, weight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("final_showdown.png", dpi=150)
    print("  → 'final_showdown.png' saved")

    # Parameter distribution for Log-SVT
    valid_results = [res for res in results_svt if res['mass'] is not None and res['mass'] > 0]
    
    if len(valid_results) > 0:
        masses = np.array([res['mass'] for res in valid_results])
        b0_m_vals = np.array([res['params'][0] for res in valid_results])
        k2_vals = np.array([res['params'][3] for res in valid_results])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # b0/m vs Mass
        axes[0].scatter(masses, b0_m_vals, alpha=0.6, c='darkred', edgecolors='k', s=40)
        axes[0].set_xscale('log')
        axes[0].set_xlabel('Baryonic Mass (M☉)', fontsize=12)
        axes[0].set_ylabel('b₀/m (km²/s²)', fontsize=12)
        axes[0].set_title('Log-SVT: Logarithmic Coupling vs Mass', fontsize=12, weight='bold')
        axes[0].grid(True, which="both", ls="--", alpha=0.3)
        if len(b0_m_vals) > 0:
            mean_b0 = np.mean(b0_m_vals)
            axes[0].axhline(mean_b0, color='r', linestyle='--', lw=2, 
                           label=f'Mean b₀/m = {mean_b0:.1f} km²/s²')
            axes[0].legend()
        
        # k2 vs Mass
        axes[1].scatter(masses, k2_vals, alpha=0.6, c='darkblue', edgecolors='k', s=40)
        axes[1].set_xscale('log')
        axes[1].set_xlabel('Baryonic Mass (M☉)', fontsize=12)
        axes[1].set_ylabel('k₂ (kpc⁻²)', fontsize=12)
        axes[1].set_title('Log-SVT: Polynomial Coefficient vs Mass', fontsize=12, weight='bold')
        axes[1].grid(True, which="both", ls="--", alpha=0.3)

        plt.tight_layout()
        plt.savefig("svt_parameter_universality.png", dpi=150)
        print("  → 'svt_parameter_universality.png' saved")

    # Statistics
    print("\n" + "=" * 60)
    print("STATISTICAL SUMMARY")
    print("=" * 60)
    
    if len(chi2_nfw) > 0:
        median_nfw = np.median(chi2_nfw)
        mean_nfw = np.mean(chi2_nfw)
        print(f"\nNFW (ΛCDM):")
        print(f"  Successful fits: {len(chi2_nfw)}/{len(files)}")
        print(f"  Median χ²ᵥ: {median_nfw:.3f}")
        print(f"  Mean χ²ᵥ: {mean_nfw:.3f}")
        print(f"  Excellent fits (χ²ᵥ < 1): {np.sum(chi2_nfw < 1)} ({100*np.sum(chi2_nfw < 1)/len(chi2_nfw):.1f}%)")
        print(f"  Good fits (χ²ᵥ < 2): {np.sum(chi2_nfw < 2)} ({100*np.sum(chi2_nfw < 2)/len(chi2_nfw):.1f}%)")
        
    if len(chi2_svt) > 0:
        median_svt = np.median(chi2_svt)
        mean_svt = np.mean(chi2_svt)
        print(f"\nLog-SVT:")
        print(f"  Successful fits: {len(chi2_svt)}/{len(files)}")
        print(f"  Median χ²ᵥ: {median_svt:.3f}")
        print(f"  Mean χ²ᵥ: {mean_svt:.3f}")
        print(f"  Excellent fits (χ²ᵥ < 1): {np.sum(chi2_svt < 1)} ({100*np.sum(chi2_svt < 1)/len(chi2_svt):.1f}%)")
        print(f"  Good fits (χ²ᵥ < 2): {np.sum(chi2_svt < 2)} ({100*np.sum(chi2_svt < 2)/len(chi2_svt):.1f}%)")

    # Head-to-head comparison
    if len(results_nfw) > 0 and len(results_svt) > 0:
        galaxies_both = set(r['galaxy'] for r in results_nfw) & set(r['galaxy'] for r in results_svt)
        nfw_wins = 0
        svt_wins = 0
        for g in galaxies_both:
            chi2_n = next((r['chi2_nu'] for r in results_nfw if r['galaxy'] == g), np.inf)
            chi2_s = next((r['chi2_nu'] for r in results_svt if r['galaxy'] == g), np.inf)
            if chi2_n < chi2_s:
                nfw_wins += 1
            else:
                svt_wins += 1
        print(f"\nHead-to-Head (N={len(galaxies_both)}):")
        print(f"  NFW wins: {nfw_wins} ({100*nfw_wins/len(galaxies_both):.1f}%)")
        print(f"  Log-SVT wins: {svt_wins} ({100*svt_wins/len(galaxies_both):.1f}%)")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
