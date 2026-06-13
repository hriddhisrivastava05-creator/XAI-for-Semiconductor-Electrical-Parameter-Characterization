# XAI for Semiconductor Electrical Parameter Characterization

Machine learning models to predict electrical parameters of proton-irradiated MOSFETs 
(SG-SOI, DG-MOSFET, GP-MOSFET), with SHAP and LIME used to interpret the physics 
governing model predictions.

📄 **Published**: IEEE VLSID 2026 — [DOI: 10.1109/VLSID68508.2026.00060](https://doi.org/10.1109/VLSID68508.2026.00060)

## Overview
Proton irradiation degrades MOSFET performance through displacement damage and 
ionization effects, which is critical for devices used in satellites and aerospace 
systems. This work predicts Gate-to-Drain Capacitance (C_GD), Gate-to-Source 
Capacitance (C_GS), and Drain Current (I_D) for SG-SOI, DG-, and GP-MOSFETs under 
varying proton fluence (1×10¹³ to 1×10¹⁵ protons/cm²), using TCAD-simulated data 
(Silvaco Victory TCAD).

## Methodology
- **Dataset**: TCAD-simulated transfer characteristics (|V_DS| = 0–2V) for n- and 
  p-channel SG-SOI, DG-, and GP-MOSFETs exposed to 1.8 MeV proton radiation at 
  5 fluence levels. Features: V_G, V_D, type, fluence, device type. Targets: C_GD, C_GS, I_D.
- **Models**: 
  - Random Forest (100 trees)
  - K-Nearest Neighbors (pipeline: predicts C_GD, C_GS → feeds into I_D prediction)
  - Ensemble MLP-RF (3 hidden layers, ReLU, soft/average voting)
- **Tasks**: Regression (C_GD, C_GS, I_D) + Classification (device type, n/p type, fluence level)
- **Validation**: 70-20-10 train-test-val split (stratified), 5-fold cross-validation, blind testing on out-of-range parameter values
- **XAI Techniques**: SHAP (global feature importance) and LIME (local, instance-specific explanations)

## Key Results
- MLP-RF achieved best regression performance (R² > 0.999 for C_GD and I_D, R² ≈ 0.98 for C_GS)
- Classification accuracy > 0.96 for device and type; fluence classification was harder (~0.81 for MLP-RF)
- SHAP and LIME consistently ranked **V_D > V_G > I_D > C_GS > C_GD** in feature importance, 
  aligning with known MOSFET device physics (channel pinch-off, inversion layer formation, Miller effect)

## Tech Stack
- Python, scikit-learn (RF, KNN, MLP)
- SHAP, LIME
- Silvaco TCAD (data generation), SRIM (damage factor / NIEL calculation)
- Matplotlib / Plotly for visualization

## Repository Structure
