# Guided Harmonic Path-Integral Diffusion (GuidedPID)

This repository contains the full Python/Jupyter implementation of **Guided Harmonic Path-Integral Diffusion (GH-PID)** and all code required to reproduce the figures and numerical experiments in the paper:

**“Generative Stochastic Optimal Transport: Guided Harmonic Path-Integral Diffusion.”**  
Michael (Misha) Chertkov, University of Arizona  
(see GuidedPID.pdf + to appear on arXiv shortly)

---

## Contents

- `CaseA_final.ipynb` — Jupyter notebook generating all figures for Cases A.
- `CaseB_final.ipynb` — Jupyter notebook generating all figures for Cases B.
- `CaseC_final.ipynb` — Jupyter notebook generating all figures for Cases C.
- `adapid_torch/` — Core Python modules (AdaPID partially adopted for pytorch)
- `CaseA/` — Core Python modules (needed to run CaseA experiments)
- `guided_torch/` — Core Python modules (needed to run Case B and Case B experiments) 


---

## Reproducing Figures

To regenerate all figures from the paper:

1. Clone the repository  
   ```bash
   git clone https://github.com/mchertkov/GuidedPID
   cd GuidedPID
