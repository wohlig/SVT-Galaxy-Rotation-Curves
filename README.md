# Superfluid Vacuum Theory: Galaxy Rotation Curves

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18150850.svg)](https://doi.org/10.5281/zenodo.18150850)

**A dark matter-free explanation for galaxy rotation curves based on quantum vacuum dynamics**

## 📄 Paper

This repository accompanies the research paper:

> **Galaxy Rotation Curves from Superfluid Vacuum Theory: A Dark Matter-Free Explanation**  
> Shah, C. (2026)  
> 🔗 **DOI:** [10.5281/zenodo.18150850](https://doi.org/10.5281/zenodo.18150850)

## Overview

This repository implements the Superfluid Vacuum Theory (SVT) model for fitting galaxy rotation curves, demonstrating that flat rotation curves emerge naturally from vacuum vortex dynamics without requiring dark matter particles.

### Key Results
- **Median χ²ᵥ = 2.53** on 175 SPARC galaxies (comparable to NFW's 2.37)
- **50.9% head-to-head wins** against standard dark matter (NFW) model
- **Only 2 free parameters** — same complexity as ΛCDM

## The SVT Velocity Formula
```
V_SVT(r) = V_∞ · √ln(1 + r/rᶜ) · [ln(1 + Mᵦ/Mᶜ)]^0.25
```

Where the "dark matter" contribution arises from **rotational kinetic energy of quantized vacuum vortices** — not invisible particles.

## Physical Basis

| Aspect | SVT | ΛCDM (NFW) |
|--------|-----|------------|
| Dark matter source | Vacuum vortex energy | Unknown particles |
| New particles required | None | Yes (undetected) |
| BTFR explanation | Natural (mass factor) | Coincidental |
| Core-cusp problem | Naturally cored | Cuspy (problematic) |

## Installation
```bash
git clone https://github.com/wohlig/SVT-SPARC-Test.git
cd SVT-SPARC-Test
pip install -r requirements.txt
```

## Usage
```python
from svt_model import V_svt
import numpy as np

# Example: Calculate SVT rotation velocity
r = np.linspace(0.1, 30, 100)  # radius in kpc
V_infty = 150  # km/s
r_c = 3.0  # kpc
M_bary = 1e10  # solar masses

v_rot = V_svt(r, V_infty, r_c, M_bary)
```

## Citation

If you use this code or find our work useful, please cite:
```bibtex
@article{shah2026svt,
  title={Galaxy Rotation Curves from Superfluid Vacuum Theory: A Dark Matter-Free Explanation},
  author={Shah, Chintan},
  year={2026},
  doi={10.5281/zenodo.18150850},
  url={https://doi.org/10.5281/zenodo.18150850}
}
```

## References

- **SPARC Database:** [Lelli, McGaugh & Schombert (2016)](https://astroweb.case.edu/SPARC/)
- **Logarithmic Quantum Mechanics:** Białynicki-Birula & Mycielski (1976)
- **Superfluid Vacuum Theory:** Volovik (2003), Zloshchastiev (2011)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Chintan Shah**  
Independent Researcher, Mumbai, India  
GitHub: [@wohlig](https://github.com/wohlig)
