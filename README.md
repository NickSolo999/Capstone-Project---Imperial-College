# Road Condition Classification using Machine Learning

**Imperial College London - Machine Learning Capstone Project**

Road surface classification (asphalt, cobblestone, dirt) and speed bump detection using inertial sensor data from the PVS dataset. KNN and SVM achieved 92% road classification accuracy. CNN achieved 90% road classification accuracy and 99% speed bump detection (with 63% recall due to class imbalance).

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methods and Results](#methods-and-results)
- [Installation](#installation)
- [Results Summary](#results-summary)
- [References](#references)

---

## Overview

**Classification Tasks:**
1. **Road Surface Type**: Asphalt, Cobblestone, Dirt
2. **Speed Bump Detection**: Binary (bump/no bump)

**Performance:**
- Traditional ML: KNN 92.2%, SVM 91-93%
- CNN with sliding windows: 90.4% 
- Speed bump: 99% accuracy but 63% recall (1:62 class imbalance)

---

## Dataset

**Source**: [PVS (Passive Vehicular Sensors) Dataset on Kaggle](https://www.kaggle.com/datasets/jefmenegazzo/pvs-passive-vehicular-sensors-datasets)

**Citation**: J. Menegazzo & A. von Wangenheim (2020). Multi-Contextual and Multi-Aspect Analysis for Road Surface Type Classification Through Inertial Sensors and Deep Learning. SBESC 2020.

### Dataset Overview
- **9 datasets**: 3 vehicles (VW Saveiro, Fiat Bravo, Fiat Palio) × 3 drivers × 3 scenarios
- **Sensors**: MPU-9250 (accelerometer, gyroscope, magnetometer, temperature) at 100 Hz
- **Placement**: Dual networks (left/right), 3 positions each: below suspension, above tire, dashboard
- **GPS**: 1 Hz (speed, location)
- **Total samples**: ~1.08M time series points
- **Features**: 50 channels (25 per side: 9 IMU axes + speed, × 3 positions)

### Class Distribution
- **Road types**: Asphalt, Cobblestone, Dirt (balanced)
- **Speed bumps**: 1.58% bumps, 98.42% no bumps (1:62 imbalance)

### Train/Test Split
- **Train**: PVS 1,2,3,7,8,9
- **Test**: PVS 4,5,6 (different vehicle contexts)

---

## Methods and Results

### 1. Traditional ML (KNN & SVM)
**Notebook**: `road_class_sup_learn.ipynb`

**Method**: Extract statistical features (mean, std, var) from 3-second windows → KNN/SVM

**Results**:
- KNN (k=5): 92.2% accuracy (F1: Asphalt 0.97, Cobblestone 0.88, Dirt 0.90)
- SVM (RBF): 91-93% accuracy (similar per-class F1)

**Pros**: No GPU, interpretable, fast training  
**Cons**: Manual features, loses temporal patterns

---

### 2. Deep Learning - Road Classification (CNN)
**Notebooks**: `road_class_CNN.ipynb`, `road_class_CNN_windowing.ipynb`

#### Sliding Window CNN 
**Method**: Raw 100-timestep windows → Multi-scale 1D CNN

**Architecture**:
- Conv1d layers with kernel sizes 100, 50, 20 (multi-scale: 1.0s, 0.5s, 0.2s patterns)
- Channels: 64 → 128 → 256
- Global pooling → Dense → 3 classes

**Training**: Adam (lr=0.001), 30 epochs, batch=256

**Performance**:
- Test Accuracy: **90.4%**
- F1 scores: Asphalt 0.96, Cobblestone 0.92, Dirt 0.92
- Inference: <10ms/batch

---

### 3. Speed Bump Detection (CNN)
**Notebook**: `speed_bump_class_CNN.ipynb`

**Challenge**: Extreme class imbalance (1.58% bumps = 1:62 ratio)

**Solutions**:
- Oversampling: Dense windowing (stride=10) in bump regions
- Class weighting: Loss weighted [1.0, 62.0]

**Results**:
- Accuracy: 99.0% (misleading - imbalance effect)
- Precision: 0.70 (70% of bump predictions correct)
- Recall: 0.63 (63% of bumps detected)
- Specificity: 0.996 (99.6% non-bumps identified)

**Interpretation**: High accuracy is deceptive due to class imbalance. Moderate precision/recall show real performance. Suitable for driver warnings, not safety-critical applications.

---

## Comparison Summary

| Approach | Road Accuracy | Speed Bump | Training Time | Inference | GPU |
|----------|---------------|------------|---------------|-----------|-----|
| KNN.     | 92.2%         | -          | None          | Slow      | No  |
| SVM      | 91.8%         | -          | Medium.       | Medium    | No  |
| CNN      | 90.4%.        | 99%        | 20min         | Fast.     | Yes |

*Speed bump: See precision/recall for true performance (63% recall)

**Key Takeaway**: Traditional ML achieves best performance and it the easiest to interpret.  CNN performance may be further improved with additional hyper paramater tuninging and different network architecture options. 

This type of model has many potential applications with ADAS and active chassis systems. One application is to classify road types for an fully active suspension (FAS) system. The classification would happen real time on an embedded chassis module and the information would be used to modify the FAS software calibration for a particular road type. For example, if the road is classified as smooth asphalt, the calibration for the FAS can be tuned to increase the Skyhook control to enable better road osolation without compromising secondary ride performance. When the classification changes to cobblestone or dirt, the calibration would reduce the Skyhook control so as not to compromise secondard ride.

Additionally, the road classification could be used to actively change driving modes. For example, switching to 'Sport' mode when driving on dry asphalt.

The Speed bump classification requires further development to improve the precision and recall. If this can be achieved, then again this type of classifier could be used for FAS system tuning. The FAS calibration could be altered when a speed bump is detected, for example reducing the wheel damping to improve the impact performance of the speed bump. Improving this model is recomended as further works as this type of classification would be extremely valuable for many active chassis systems. 

---

## Installation

```bash
# Clone and setup
git clone https://github.com/NickSolo999/Imperial-College-Capstone.git
cd Imperial-College-Capstone
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn torch jupyter joblib

# Download PVS dataset from Kaggle and place in data/ folder
```

---

## Documentation

- **[data_sheet.md](data_sheet.md)**: Dataset documentation (motivation, composition, collection, preprocessing, uses, distribution)
- **[model_card.md](model_card.md)**: Model architectures, performance, limitations, trade-offs

---

## References

1. **Dataset**: Menegazzo, J., & von Wangenheim, A. (2020). Multi-Contextual and Multi-Aspect Analysis for Road Surface Type Classification Through Inertial Sensors and Deep Learning. SBESC 2020. https://doi.org/10.1109/SBESC51047.2020.9277846

2. **Kaggle**: https://www.kaggle.com/datasets/jefmenegazzo/pvs-passive-vehicular-sensors-datasets/data

3. **License**: CC BY-NC-ND 4.0 (dataset), MIT (code)

---

## License

This project is for academic purposes (Imperial College London ML Capstone). PVS dataset licensed under CC BY-NC-ND 4.0.

---

**Last Updated**: October 2025
