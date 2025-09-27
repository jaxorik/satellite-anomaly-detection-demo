# Black Box Algorithm vs. Interpretable Model

## A Demonstration of Reproducible Models on Complex Datasets

**Created by Rae Chipera**  
**Jaxorik AI Research Group**, registered in SAM.GOV for federal contracting

- **UEID**: K8ENCCGZ2M13
- **Cage Code**: 11X70
- **Disaster Response registry**: Active
- **Eligible for WOSB and SDVOSB set-asides**

**FOR CONTRACT INQUIRIES**: govinquiries@jaxorik.com

---

## Overview

This demonstration compares black box deep learning approaches with transparent machine learning models for satellite imagery anomaly detection. Using realistic synthetic multispectral data, we show that interpretable models can match or exceed black box performance while providing critical explainability for high-stakes decision-making.

## Why This Matters for Defense & Disaster Response

In mission-critical applications, understanding *why* a model flagged something as anomalous is as important as the detection itself. Transparent models enable:

- **Operational trust**: Analysts can validate detections before taking action
- **Reduced false positives**: Understanding feature contributions helps filter noise
- **Accountability**: Clear decision paths for after-action review
- **Rapid adaptation**: Interpretable rules can be adjusted for new threat types

## What's Demonstrated

- **Realistic satellite imagery simulation**: 12 spectral/spatial features across multiple bands (RGB, NIR, SWIR, Thermal, computed indices)
- **Defense-relevant anomaly types**: New construction, camouflaged structures, fires, flooding, vehicle concentrations
- **Three model approaches**: 
  - Autoencoder (black box)
  - Decision Trees (transparent)
  - Reservoir Computing (transparent)
- **Performance comparison**: Accuracy, ROC-AUC, and per-anomaly-type detection rates
- **Explainability analysis**: Side-by-side comparison of what each model can tell you about its decisions

## Results Summary

The transparent models achieved **99%+ accuracy** with perfect AUC scores, outperforming the black box autoencoder (84% accuracy) while providing full explainability.

| Model | Accuracy | AUC-ROC | Explainability |
|-------|----------|---------|----------------|
| Autoencoder (Black Box) | 0.838 | 0.996 | ❌ None |
| Decision Tree | 0.998 | 0.999 | ✅ Rule-based paths |
| Reservoir Computing | 0.997 | 1.000 | ✅ Feature contributions |

## Installation

```bash
git clone https://github.com/jaxorik/satellite-anomaly-detection-demo.git
cd satellite-anomaly-detection-demo
pip install -r requirements.txt
```

## Usage

### Run the Full Demo

```bash
python main.py
```

This will:
1. Generate realistic synthetic satellite data (64x64 grid, 12 features)
2. Train all three models
3. Generate comprehensive visualizations
4. Output performance metrics and explainability comparisons

### Explore the Jupyter Notebook

```bash
jupyter notebook demo.ipynb
```

The notebook includes detailed explanations and allows you to experiment with parameters.

## Generated Outputs

The demo generates the following visualizations:

- `spectral_bands.png` - All 12 spectral/spatial features
- `rgb_composites.png` - True color and false-color band combinations
- `model_predictions_comparison.png` - Side-by-side model performance
- `autoencoder_details.png` - Detailed analysis of black box model
- `decision_tree_details.png` - Decision tree analysis and visualization
- `reservoir_details.png` - Reservoir computing analysis
- `performance_metrics.png` - ROC curves, feature importance, confusion matrices
- `decision_tree_full.png` - Full decision tree visualization

## Repository Structure

```
satellite-anomaly-detection-demo/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── main.py                     # Main execution script
├── demo.ipynb                  # Jupyter notebook with explanations
├── data_generation.py          # Realistic satellite data generator
├── models.py                   # Model implementations
├── visualization.py            # Visualization functions
└── LICENSE                     # MIT License
```

## Key Features

### Realistic Data Generation

The synthetic data includes:
- **10 spectral bands**: RGB, NIR, SWIR (2 bands), Thermal (2 bands), NDVI, NDWI
- **2 spatial features**: Edge magnitude, texture variance
- **Spatially correlated terrain**: Vegetation, water, urban areas with realistic transitions
- **5 anomaly types with realistic signatures**:
  - Construction: High reflectance, geometric edges, low vegetation
  - Camouflage: Visible/NIR spectral mismatch
  - Fires: Extreme thermal signature with smoke
  - Flooding: Water signature in dry areas
  - Vehicles: Small bright spots with thermal signatures

### Model Implementations

1. **Autoencoder (Black Box)**
   - Neural network trained on reconstruction error
   - High performance but no explainability
   - Demonstrates the "trust me" problem

2. **Decision Tree**
   - Clear rule-based decision paths
   - Visualizable structure
   - Easy to validate and modify

3. **Reservoir Computing**
   - Feature-level contribution analysis
   - Balance between complexity and interpretability
   - Novel approach to transparent ML

## Requirements

- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy

See `requirements.txt` for exact versions.

## Technical Documentation

For detailed technical documentation and white paper, visit [jaxorik.com](https://jaxorik.com).

## Citation

If you use this code in your research or applications, please cite:

```
Chipera, R. (2025). Black Box vs. Interpretable Models: 
Satellite Anomaly Detection Demo. Jaxorik AI Research Group.
https://github.com/jaxorik/satellite-anomaly-detection-demo
```

## License

MIT License - See LICENSE file for details.

## Contact

For technical questions or collaboration opportunities:
- Email: govinquiries@jaxorik.com
- Website: [jaxorik.com](https://jaxorik.com)

---

**Note**: This demonstration uses synthetic data for illustration purposes. For production deployments with real satellite imagery, contact us for customized solutions tailored to your specific operational requirements.
