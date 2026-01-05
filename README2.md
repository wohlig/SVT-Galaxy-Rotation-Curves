# SPARC Data JSON Converter

This repository contains the [SPARC (Spitzer Photometry and Accurate Rotation Curves)](http://astroweb.cwru.edu/SPARC/) database converted into JSON format for easy usage in Jupyter Notebooks and other analysis tools.

## Structure

- `sparc_data/`: Contains the original `.dat` files from the SPARC database.
- `sparc_full.json`: Contains the aggregated data for all galaxies in a single file.
- `convert_to_json.py`: Python script used to perform the conversion.

## JSON Format

Each JSON file represents a galaxy and follows this structure:

```json
{
  "meta": {
    "galaxy_name": "GalaxyName",
    "distance": 12.3,
    "distance_unit": "Mpc"
  },
  "columns": ["Rad", "Vobs", "errV", "Vgas", "Vdisk", "Vbul", "SBdisk", "SBbul"],
  "units": ["kpc", "km/s", "km/s", "km/s", "km/s", "km/s", "L/pc^2", "L/pc^2"],
  "data": [
    [0.1, 10.5, 2.1, ...],
    [0.2, 20.3, 1.5, ...]
  ]
}
```

## Models Tested

### 1. Lambda CDM (NFW Halo)
Standard dark matter halo profile.
- **Formula**: $V_{NFW}^2(r) = V_{char}^2 \frac{r_s}{r} \left[ \ln(1+r/r_s) - \frac{r/r_s}{1+r/r_s} \right]$
- **Free Parameters**: 
    - $V_{char}$: Characteristic velocity
    - $r_s$: Scale radius

### 2. SVT (Superfluid Vacuum Theory)
Logarithmic velocity profile derived from a superfluid vacuum with a logarithmic equation of state ($P = b\rho$).
- **Formula**: $V(r) = V_\infty \sqrt{\ln(1 + r/r_c)} \times [\ln(1 + M_{bary}/M_{crit})]^{0.25}$
- **Free Parameters**:
    - $V_\infty$: Asymptotic velocity scale
    - $r_c$: Core radius of the vortex
- **Fixed Parameter**:
    - $M_{crit} = 10^6 M_\odot$

## Columns Description

- **Rad**: Radius (kpc)
- **Vobs**: Observed rotation velocity (km/s)
- **errV**: Uncertainty in observed velocity (km/s)
- **Vgas**: Velocity contribution from gas (km/s)
- **Vdisk**: Velocity contribution from the stellar disk (km/s)
- **Vbul**: Velocity contribution from the bulge (km/s)
- **SBdisk**: Surface brightness of the disk (L/pc^2)
- **SBbul**: Surface brightness of the bulge (L/pc^2)

## Usage
    
You can load the entire database easily in Python:

```python
import json
import pandas as pd

# Load all data
with open('sparc_full.json', 'r') as f:
    all_data = json.load(f)

# Access a specific galaxy
galaxy_name = 'CamB'
data = all_data[galaxy_name]

# Access metadata
print(f"Galaxy: {data['meta']['galaxy_name']}, Distance: {data['meta']['distance']} Mpc")

# Convert to Pandas DataFrame
df = pd.DataFrame(data['data'], columns=data['columns'])
print(df.head())
```

## Regeneration

To regenerate the JSON files from the source data:

```bash
python3 convert_to_json.py
```
