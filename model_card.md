# Model Card: Road Condition Classification Models

ML models for road surface classification and speed bump detection using the PVS (Passive Vehicular Sensors) dataset.

---

## Model 1: Traditional ML (KNN & SVM)

### Model Description

**Input:** Statistical features from 3-second sensor windows (300 samples)
- Features: mean, std, var per sensor channel
- 50 sensor channels: accelerometer, gyroscope, magnetometer, speed (left/right phones)
- 150 features after aggregation

**Output:** 3 classes - Asphalt (0), Cobblestone (1), Dirt (2)

**Algorithms:**
- **KNN**: k=5, Euclidean distance, majority vote
- **SVM**: RBF kernel, C=0.1, gamma='scale', One-vs-Rest

### Performance

**Split:** Train on PVS 1,2,3,7,8,9 / Test on PVS 4,5,6

**Results:**
- KNN: 92.2% accuracy (F1: Asphalt 0.97, Cobblestone 0.88, Dirt 0.90)
- SVM: 91.8% accuracy (similar F1 scores)

**Observations:**
- Asphalt easiest to classify
- Some confusion between cobblestone and dirt

### Limitations

1. **Performance:** KNN slow inference, SVM memory-intensive
2. **Feature Engineering:** Requires manual feature extraction
3. **Temporal Loss:** Statistical aggregation loses sequential patterns
4. **Scalability:** KNN O(n) inference complexity
5. **Generalization:** Device-dependent (specific phone models/positions)

### Trade-offs

**KNN:**
- ✅ Simple, no training, multi-class ready
- ❌ Slow inference, memory-intensive, sensitive to scaling

**SVM:**
- ✅ High-dimensional effective, memory-efficient (support vectors only)
- ❌ Requires tuning, slow training on large datasets

---

## Model 2: Deep Learning - Road Classification (CNN)

### Model Description

**Input:** Raw time series in sliding windows
- Window: 100 timesteps (1 second at 100 Hz), stride=50 (50% overlap)
- Shape: (batch, 50 features, 100 timesteps)
or
- Statistical features from 3-second sensor windows (300 samples)
- Features: mean, std, var per sensor channel


**Output:** 3 classes (Asphalt, Cobblestone, Dirt)

**Architecture (sliding window):**
```
Conv1d: 50→64 channels, kernel=100 (1.0s patterns) + ReLU + BatchNorm + MaxPool + Dropout(0.15)
Conv1d: 64→128 channels, kernel=50 (0.5s patterns) + ReLU + BatchNorm + MaxPool + Dropout(0.2)
Conv1d: 128→256 channels, kernel=20 (0.2s patterns) + ReLU + BatchNorm + Dropout(0.25)
GlobalAvgPool → Dense(128) → Dropout(0.3) → Dense(3) + Softmax
```

**Training:** Adam optimizer (lr=0.001), 30 epochs, batch=256, cross-entropy loss

### Performance

**Results:**
- Test Accuracy - sliding window (1s): 88.1%
============================================================
Per-Class Metrics:
------------------------------------------------------------
Class           Precision    Recall       F1-Score    
------------------------------------------------------------
Asphalt         0.969        0.959        0.964       
Cobblestone     0.812        0.814        0.813       
Dirt            0.834        0.844        0.839 

- Test Accuracy - statistical features (3s): 90.4%
============================================================
Per-Class Metrics:
------------------------------------------------------------
Class           Precision    Recall       F1-Score    
------------------------------------------------------------
Asphalt         0.993        0.992        0.993       
Cobblestone     0.842        0.835        0.838       
Dirt            0.846        0.853        0.850 

### Limitations

1. **Computational:** Requires GPU, longer training
2. **Data:** Needs more samples than traditional ML
3. **Interpretability:** Black box, harder to explain decisions
4. **Generalization:** May overfit without proper regularization
5. **Device Dependency:** Performance varies with sensor quality

### Trade-offs

✅ **Advantages:**
- Highest accuracy
- Automatic feature extraction (sliding window method)
- Fast inference after training
- Multi-scale temporal pattern recognition

❌ **Disadvantages:**
- GPU required
- Longer training time
- Less interpretable
- More data-hungry
  - Dropout: 0.25

Global Average Pooling:
  - AdaptiveAvgPool1d: 25 → 1 timestep
  
Fully Connected Layers:
  - Linear: 256 → 128, ReLU, Dropout 0.3
  - Linear: 128 → 3 (output classes)
  
Output: Softmax probabilities over 3 classes
```

**Total Parameters:** ~150,000-200,000 trainable parameters

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss
- Batch size: 256
- Epochs: 30
- Device: MPS (Apple Silicon GPU) / CUDA / CPU
---

## Model 3: Speed Bump Detection (CNN)

### Model Description

**Input:** Raw time series in sliding windows
- Window: 100 timesteps (1 second), stride=50 (normal), stride=10 (bump regions - oversampling)
- Shape: (batch, 50 features, 100 timesteps)

**Output:** Binary classification (No bump = 0, Speed bump = 1)

**Architecture:** Similar to road CNN, binary output, class weights [1.0, 62.0]

**Training:** Adam (lr=0.001), CrossEntropyLoss with class weights, batch=64, 30 epochs

### Performance

**Results:**
- Accuracy: 99.0% (misleading - imbalance effect)
- **Precision: 0.70** (70% bump predictions correct)
- **Recall: 0.63** (63% bumps detected)
- Specificity: 0.996 (99.6% non-bumps identified)

**Class Imbalance:** 98.42% no bump, 1.58% bump (1:62 ratio)

**Confusion Matrix:**
```
Predicted:      No Bump    Bump
Actual No Bump:   7108      30
Actual Bump:        41      69
```

### Limitations

1. **Class Imbalance:** High accuracy misleading, precision/recall more meaningful
2. **False Negatives:** 37% bumps missed
3. **False Positives:** 30% false alarm rate
4. **Generalization:** Trained on specific bump types (asphalt, cobblestone)
5. **Latency:** 100ms window introduces delay

### Trade-offs

**Imbalance Solutions:**
- Class weighting: Simple but unstable
- Oversampling: Better learning but overfitting risk

**Threshold Tuning:**
- Default (0.5): Balanced
- Lower: Higher recall, more false alarms
- Higher: Fewer false alarms, miss more bumps

**Use Cases:**
- ✅ Driver warning systems
- ✅ Road monitoring (aggregate stats)
- ❌ Safety-critical control (insufficient recall)

---

## Model Comparison & Recommendations

### Selection Guide

| Need | Recommended Model | Reason |
|------|------------------|---------|
| Best accuracy | KNN | 92% |
| Interpretability | KNN/SVM | Understandable decisions |
| Limited GPU | SVM | CPU-friendly, 92% accuracy |
| Fast deployment | SVM/CNN | Moderate complexity |
| Speed bump detection | CNN with tuning | Handle imbalance, but test thoroughly |

### Future Improvements

1. **Models:** LSTM/GRU for temporal sequences, attention mechanisms, ensemble methods
2. **Data:** SMOTE, time series augmentation, cross-device collection
3. **Optimization:** Quantization (FP32→INT8), pruning, ONNX export
4. **Evaluation:** Cross-validation, real-world testing, calibration analysis

---

## References

- J. Menegazzo & A. von Wangenheim (2020). Multi-Contextual and Multi-Aspect Analysis for Road Surface Type Classification Through Inertial Sensors and Deep Learning. SBESC 2020.
- Dataset: https://www.kaggle.com/datasets/jefmenegazzo/pvs-passive-vehicular-sensors-datasets
