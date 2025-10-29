# PVS Dataset - Datasheet

## Motivation

**Purpose:** Research dataset for road surface classification and speed bump detection using passive inertial sensors (accelerometer, gyroscope, magnetometer). Applications: infrastructure monitoring, vehicle safety, autonomous driving.

**Creators:** J. Menegazzo & A. von Wangenheim (Federal University of Santa Catarina, Brazil)

**Citation:**
```
J. Menegazzo and A. von Wangenheim, "Multi-Contextual and Multi-Aspect Analysis for Road Surface 
Type Classification Through Inertial Sensors and Deep Learning," SBESC 2020, pp. 1-8, 
doi: 10.1109/SBESC51047.2020.9277846
```

**License:** CC BY-NC-ND 4.0 (Attribution-NonCommercial-NoDerivatives)

---

## Composition

**Data Type:** Time series sensor readings from MPU-9250 IMUs mounted in vehicles

**Instance Components:**
- **Inertial sensors:** 3-axis accelerometer, gyroscope, magnetometer, temperature (100 Hz)
- **GPS:** Location, speed, heading (1 Hz)
- **Video:** External environment camera (720p, 30 Hz)
- **Labels:** Road surface types (asphalt, cobblestone, dirt), speed bumps, road quality

**Dataset Statistics:**
- **9 PVS datasets:** 3 vehicles (VW Saveiro, Fiat Bravo, Fiat Palio) × 3 drivers × 3 scenarios
- **Total samples:** ~1.08M time-series samples
- **Sampling rate:** 100 Hz
- **Features:** 50 total (25 per side: 9 IMU axes + speed, × 3 positions per side)
- **Sensor placement:** Dual networks (left/right), 3 modules each (control arm, above tire, dashboard)

**Class Distribution:**
- Road surfaces: Asphalt, cobblestone, dirt (balanced)
- Speed bumps: 1.58% positive (1:62 imbalance)
- Road quality: Good, regular, bad (per side)

**Missing Data:** Some GPS dropouts, handled via interpolation or zero-filling during preprocessing

---

## Collection Process

**Hardware:**
- MPU-9250 modules (accelerometer, gyroscope, magnetometer, temperature)
- HP Webcam HD-4110 (720p, 30 Hz)
- Xiaomi Mi 8 GPS (1 Hz)

**Mounting:** 
- Camera: External roof
- GPS: Dashboard
- 2 sensor networks (left/right), 3 modules each:
  - Control arm (below suspension)
  - Above tire (body)
  - Dashboard (cabin)

**Collection Context:**
- 3 vehicles, 3 drivers, 3 scenarios
- Real roads with asphalt, cobblestone, dirt surfaces
- Various speed bumps and road quality levels
- Each session: 10-15 km, ~10-30 minutes

**Time Frame:** Data collected 2019-2020 (publication: 2020)

---

## Preprocessing/Cleaning/Labelling

**Applied Preprocessing:**
1. **Normalization:** StandardScaler (zero mean, unit variance)
2. **Feature Selection:** Retained accelerometer, gyroscope, magnetometer, speed
3. **Label Encoding:** 
   - Road: One-hot → categorical (0=asphalt, 1=cobblestone, 2=dirt)
   - Speed bump: Binary (0=no bump, 1=bump)
4. **Missing Values:** Filled with zeros post-normalization
5. **Windowing (Deep Learning):**
   - Sliding windows: 100 samples (1 second), stride=50 (50% overlap)
   - Oversampling: stride=10 for speed bumps (class imbalance mitigation)
6. **Statistical Features (Traditional ML):** Mean, std, min, max, median per window

**Raw Data:** Preserved in `data/` directory for alternative preprocessing

---

## Uses

**Applications:**
- Road infrastructure monitoring
- Smart city road quality mapping
- Autonomous vehicle road assessment
- Adaptive suspension systems
- Driver assistance (speed bump warnings)
- Navigation route optimization

**Limitations:**
- Device-dependent (smartphone sensor variations)
- Fixed mounting positions required
- Speed-dependent performance
- Geographic transferability varies (road types differ by region)
- Class imbalance (speed bumps: 1.58%)
- Sensor drift over time

**Inappropriate Uses:**
- Safety-critical systems without extensive validation
- Cross-device deployment without recalibration
- Real-time systems without latency validation

---

## Distribution

**Access:** https://www.kaggle.com/datasets/jefmenegazzo/pvs-passive-vehicular-sensors-datasets  
**License:** CC BY-NC-ND 4.0 (non-commercial, no derivatives)  
**Citation Required:** See above

---

## Maintenance

**Maintainers:** J. Menegazzo & A. von Wangenheim  
**Version:** Static dataset (v1, 2020)  
**Contact:** Via Kaggle dataset page or published research

---

## Ethical Considerations

**Privacy:** GPS coordinates included, but no personal identifying information. Research setting with appropriate consent.

**Bias:** Geographic-specific (Brazilian roads). May not generalize globally due to regional road characteristics, vehicle types, sensor placements.

**Impact:** Positive societal benefit - improved road safety and infrastructure monitoring.
