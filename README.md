# Superfluid Vacuum Theory: Galaxy Rotation Curves

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**A dark matter-free explanation for galaxy rotation curves based on quantum vacuum dynamics**

## Overview

This repository implements the Superfluid Vacuum Theory (SVT) model for fitting galaxy rotation curves, demonstrating that flat rotation curves emerge naturally from vacuum vortex dynamics without requiring dark matter particles.

### Key Results
- **Median χ²ᵥ = 2.53** on 175 SPARC galaxies (comparable to NFW's 2.37)
- **50.9% head-to-head wins** against standard dark matter (NFW) model
- **Only 2 free parameters** — same complexity as ΛCDM

## The SVT Velocity Formula
```
V_SVT(r) = V_∞ · √ln(1 + r/rₖ) · [ln(1 + Mₓ/Mₓ)]^0.25
```

Where the "dark matter" contribution arises from **rotational kinetic energy of quantized vacuum vortices** — not invisible particles.

## Physical Basis

| Aspect | SVT | ΛCDM (NFW) |
|--------|-----|------------|
| Dark matter source | Vacuum vortex energy | Unknown particles |
| New particles required | None | Yes (undetected) |
| BTFR explanation | Natural (mass factor) | Coincidental |
| Core-cusp problem | Naturally cored | Cuspy (problematic) |

## Citation

If you use this code, please cite:
> Shah, C. (2026). Galaxy Rotation Curves from Superfluid Vacuum Theory: A Dark Matter-Free Explanation.

## References

- SPARC Database: [Lelli, McGaugh & Schombert (2016)](https://astroweb.case.edu/SPARC/)
- Logarithmic Quantum Mechanics: Białynicki-Birula & Mycielski (1976)
- Superfluid Vacuum Theory: Volovik (2003), Zloshchastiev (2011)