# Logarithmic Superfluid Vacuum Theory: Galaxy Rotation Curves

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**An implementation of K.G. Zloshchastiev's logarithmic superfluid vacuum theory for galaxy rotation curves — modeling the physical vacuum as a Bose-Einstein condensate governed by the Log-NLSE.**

---

## 📄 Paper

This repository implements the theoretical framework from:

> **Galaxy rotation curves in superfluid vacuum theory**  
> Zloshchastiev, K.G. (2022)  
> 🔗 [ResearchGate](https://www.researchgate.net/publication/366232848_Galaxy_rotation_curves_in_superfluid_vacuum_theory)

---

## 🌌 Overview

This repository implements a **dark matter-free explanation** for galaxy rotation curves based on **Logarithmic Superfluid Vacuum Theory**. The core idea:

> **The physical vacuum is a Bose-Einstein condensate described by the logarithmic nonlinear Schrödinger equation (Log-NLSE), which induces an effective gravitational potential with multiple scale components.**

The theory produces gravity with **Newtonian, logarithmic, linear (Rindler), and quadratic (de Sitter)** terms — explaining flat rotation curves without invisible particles.

---

## 🔬 The Physics

### Log-NLSE (Fundamental Equation)

The vacuum wavefunction ψ satisfies:

$$i\hbar \frac{\partial \psi}{\partial t} = -\frac{\hbar^2}{2m}\nabla^2\psi + m\Phi\psi - b\ln\left(\frac{|\psi|^2}{\rho_0}\right)\psi$$

Key property: The **speed of sound is density-independent** ($c_s = \sqrt{b/m}$), providing a natural explanation for the constancy of the speed of light.

### Effective Gravitational Potential

From the Log-NLSE, the induced potential has multiple scale components:

$$\Phi = \Phi_{smi}(r) + \Phi_{RN}(r) + \Phi_{N}(r) + \Phi^{(ln)}(r) + \Phi^{(1)}(r) + \Phi^{(2)}(r)$$

| Term | Formula | Physical Meaning |
|------|---------|------------------|
| **Sub-Microscopic** | $\Phi_{smi} \propto -r^{-2}\ln(r)$ | Near-singularity behavior |
| **Reissner-Nordström** | $\Phi_{RN} \propto -r^{-2}$ | Charge-like effect |
| **Newtonian** | $\Phi_N = -GM/r$ | Standard gravity |
| **Logarithmic** | $\Phi^{(ln)} \propto \ln(r)$ | Flat rotation curves |
| **Rindler (Linear)** | $\Phi^{(1)} \propto r$ | Galactic acceleration |
| **de Sitter (Quadratic)** | $\Phi^{(2)} \propto -r^2$ | Cosmological expansion |

### Rotation Velocity Formula

The rotation velocity from the truncated potential (eq. 18 in paper):

$$v(R) = \sqrt{v_N^2 + \frac{b_0}{m}R\frac{d}{dR}\left[\chi\ln\left(\frac{R}{\bar{\ell}}\right) + \ln\left[(k_2R^2 + k_1R + 1)^2\right]\right] + \tilde{a}_1 R - \tilde{a}_2 R^2}$$

Which simplifies to:

$$v(R) = \sqrt{v_N^2 + \frac{b_0}{m}\left[\chi + \frac{4k_2R^2 + 2k_1R}{k_2R^2 + k_1R + 1}\right] + \tilde{a}_1 R - \tilde{a}_2 R^2}$$

---

## 📊 Key Results

### SPARC Database Validation (This Repository)

Tested against **175 galaxies** from the SPARC database:

| Metric | Log-SVT | NFW (ΛCDM) |
|--------|---------|------------|
| **Median χ²ᵥ** | **1.48** | 2.87 |
| **Excellent fits (χ²ᵥ < 1)** | **40.6%** | 26.9% |
| **Good fits (χ²ᵥ < 2)** | **55.8%** | 41.1% |
| **Head-to-head wins** | **56.4%** | 43.6% |

### Visual Results

### Model Comparison
![Model Comparison](final_showdown.png)

### SVT Parameter Distributions
![SVT Parameter Distributions](svt_parameter_universality.png)

---

## 🏆 Winners Circle: Where Log-SVT Outperforms ΛCDM

Below are the top 5 cases where Log-SVT wins by a significant margin.

### Analysis of the Top Performers

#### 1. IC2574 (The Clear Winner)

![IC2574 Rotation Curve](galaxy_fits/IC2574_fit.png)

The most dramatic example in the dataset. The NFW profile fails significantly, while SVT provides an excellent fit.
- **$\chi^2_{NFW}$:** 42.68
- **$\chi^2_{SVT}$:** 2.57
- **The Margin:** SVT is approximately **16x better**.
- **Context:** IC2574 is a faint dwarf galaxy. These types of galaxies (Core-Cusp problem) are notoriously difficult for NFW to fit, making them prime candidates for the Log-SVT model.

#### 2. NGC3109 (The "Perfect" Fit)

![NGC3109 Rotation Curve](galaxy_fits/NGC3109_fit.png)

The NFW fit is poor, but the SVT fit is exceptionally precise.
- **$\chi^2_{NFW}$:** 11.11
- **$\chi^2_{SVT}$:** 0.17
- **The Margin:** SVT is roughly **65x better**.
- **Context:** The Log-SVT model fits the rotation curve almost perfectly ($\chi^2 \ll 1$ implies an extremely close match to data points).

#### 3. DDO154 (Significant Recovery)

![DDO154 Rotation Curve](galaxy_fits/DDO154_fit.png)

A gas-rich dwarf galaxy often used as a benchmark in dark matter studies.
- **$\chi^2_{NFW}$:** 16.67
- **$\chi^2_{SVT}$:** 1.83
- **The Margin:** SVT is roughly **9x better**.
- **Context:** NFW provides a statistically rejected fit, whereas SVT brings the fit down to a physically plausible range.

#### 4. D631-7

![D631-7 Rotation Curve](galaxy_fits/D631-7_fit.png)

A strong technical win where NFW is clearly disfavored.
- **$\chi^2_{NFW}$:** 10.15
- **$\chi^2_{SVT}$:** 1.91
- **The Margin:** SVT is roughly **5x better**.
- **Context:** The SVT model successfully reduces a high error margin down to a standard acceptable fit level.

#### 5. NGC0055

![NGC0055 Rotation Curve](galaxy_fits/NGC0055_fit.png)

SVT resolves the poor NFW fit nicely.
- **$\chi^2_{NFW}$:** 6.07
- **$\chi^2_{SVT}$:** 1.75
- **The Margin:** SVT is roughly **3.5x better**.
- **Context:** This represents a "clean" win where the alternative model provides a stable solution where NFW produces tension with the data.

### Summary Table of the Winners

| Galaxy | $\chi^2$ NFW (Standard) | $\chi^2$ Log-SVT (Alternative) | Improvement Factor |
|--------|--------------------------|--------------------------------|--------------------|
| **IC2574** | 42.68 | 2.57 | ~16.6x |
| **NGC3109** | 11.11 | 0.17 | ~65.3x |
| **DDO154** | 16.67 | 1.83 | ~9.1x |
| **D631-7** | 10.15 | 1.91 | ~5.3x |
| **NGC0055** | 6.07 | 1.75 | ~3.5x |

> [!NOTE]
> The Log-SVT model performs exceptionally well in **Dwarf and Irregular galaxies** (like IC2574 and DDO154) where the ΛCDM "cusp" usually conflicts with the observed "cores" (the Core-Cusp Problem).

---

## ✨ Physical Insights from the Paper

1. **Logarithmic term dominance:** The b₀/m parameter produces flat rotation curve (FRC) regimes across large galactic distances.

2. **χ parameter:** Usually zero, except for galaxies whose velocity doesn't approach small values at R→0 (e.g., NGC 925, 2841).

3. **Linear/Rindler term:** Affects middle-to-far regions, but often improves fits when omitted (may reflect disk model uncertainty).

4. **Quadratic/de Sitter term:** Contributes to asymptotic behavior at galactic borders; improves fitting at largest radii.

5. **Galaxy-dependent parameters:** Both linear and quadratic terms are unique to each galaxy, supporting multiple expansion mechanisms.

---

## 📁 Repository Structure

```
Log-SVT-Galaxy-Rotation/
├── README.md                        # This file
├── Paper.pdf                        # Zloshchastiev's paper
├── model_fitter.py                  # Main fitting code
├── galaxy_chi2_comparison.csv       # Results for all 175 galaxies
├── final_showdown.png               # Model comparison histogram
├── svt_parameter_universality.png   # Parameter distributions
├── galaxy_fits/                     # Individual galaxy plots
│   └── [galaxy_name]_fit.png
└── sparc_data/
    └── sparc_database/              # SPARC galaxy data files
```

---

## 🚀 Installation

```bash
git clone https://github.com/wohlig/SVT-Galaxy-Rotation-Curves.git
cd SVT-Galaxy-Rotation-Curves
pip install numpy scipy matplotlib pandas
```

---

## 💻 Usage

### Running the Model Fitter

```bash
python model_fitter.py
```

This will:
1. Load all 175 SPARC galaxy rotation curves
2. Fit both NFW (ΛCDM) and Log-SVT models
3. Generate comparison plots in `galaxy_fits/`
4. Output statistics to `galaxy_chi2_comparison.csv`

---

## 📚 References

- **Paper:** Zloshchastiev, K.G. (2022). [Galaxy rotation curves in superfluid vacuum theory](https://www.researchgate.net/publication/366232848)
- **Log-NLSE Theory:** 
  - Zloshchastiev, K.G. *Universe* 6, 180 (2020)
  - Zloshchastiev, K.G. *Grav. Cosmol.* 16, 288-297 (2010)
- **SPARC Database:** [Lelli, McGaugh & Schombert (2016)](https://astroweb.case.edu/SPARC/)
- **THINGS Data:** Walter et al., *Astron. J.* 136, 2563-2647 (2008)
- **Logarithmic Quantum Mechanics:** Białynicki-Birula & Mycielski (1976)

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 👤 Author

Implementation by: **Chintan Shah**  
📧 chintan@wohlig.com

Original theory: **K.G. Zloshchastiev**

---

<p align="center">
  <i>"The missing mass is not missing particles — it is the gravitating energy of the quantum vacuum's logarithmic configuration."</i>
</p>
