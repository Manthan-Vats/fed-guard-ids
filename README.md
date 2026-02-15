# Fed-Guard-IDS  
## Byzantine-Robust Federated Intrusion Detection System
**Trust-Weighted Geometric Median Aggregation with Honeypot-Based Behavioral Detection**

---

## Overview
Fed-Guard-IDS is a research-oriented implementation of a Byzantine-robust federated learning framework for IoT intrusion detection.

The system evaluates multiple aggregation strategies under adversarial (Byzantine) settings and introduces a trust-weighted geometric median approach (Trust-GM) incorporating honeypot-based behavioral scoring.

This repository contains:
- The complete experimental notebook `notebooks/fed_guard_ids_v1.ipynb`)
- Experiment outputs `results/`)
- A minimal reproducibility script `reproduce.ps1`)
- Dependency specification `requirements.txt`)
- Future modularization scaffolding `src/`)

---

## Implemented Components (Validated From Notebook)

### Federated Learning Setup
- 9 federated clients (device IDs 1–9)
- 30 FL rounds
- Deterministic seed = 42
- Train/Validation/Test split = 70% / 15% / 15%

### Model Architecture (IDSModel)
Multilayer Perceptron:
115 → 128 → 64 → 32 → 2  
ReLU activations  
Dropout regularization  

### Aggregation Methods Implemented
- FedAvg
- Krum
- Geometric Median
- Trust-Weighted Geometric Median (Trust-GM)

### Adversarial Simulation
- Label-flipping Byzantine clients
- Honeypot-based trust scoring
- Trust decay and dynamic weighting

### Preprocessing
- StandardScaler (fit on training data)
- SMOTE class balancing
- Feature dimension: 115

---

## Dataset
Target Dataset: **N-BaIoT (Network-based IoT Botnet Dataset)**

The notebook expects the dataset in Kaggle format by default.  
For local execution, place dataset CSV files inside:
```
data/raw/
```

The dataset is NOT included in this repository.

---

## Repository Structure
```
fed-guard-ids/
├── notebooks/ # Full experimental notebook
├── src/ # Future modularized implementation
├── results/ # Generated CSV metrics + figures
├── docs/ # GitHub Pages presentation assets
├── requirements.txt
├── reproduce.ps1
└── README.md
```

---

## Results
All experimental outputs are stored in:
```
results/all_results.csv
results/summary_table.csv
results/figures/
```

Refer to these files for authoritative performance metrics.

This README intentionally does not hardcode numeric claims to ensure consistency with the actual result files.

---

## Running (Optional — Not Required for Viewing)
If you want to run locally in the future:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook notebooks/fed_guard_ids_v1.ipynb
```

Then execute all cells.

---

## Security Notice
This repository does NOT include:
- Raw dataset files
- API keys
- kaggle.json
- Model checkpoints

Ensure .gitignore prevents committing such files.

---

## Citation
```
@misc{vats2026fedguard,
  title={Fed-Guard-IDS: Byzantine-Robust Federated Intrusion Detection with Honeypot Trust Scoring},
  author={Vats, Manthan},
  year={2026},
  institution={Manipal University Jaipur}
}
```

---

## License
MIT License — see LICENSE file.

---

## Author
Manthan Vats  
Computer Science & Engineering  
Manipal University Jaipur