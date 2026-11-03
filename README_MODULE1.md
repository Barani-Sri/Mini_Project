# Module 1: CirCor PCG Standardization & Fixed-Window Pipeline

Module 1 converts raw phonocardiogram (PCG) acoustic recordings from the **CirCor DigiScope Phonocardiogram Dataset (v1.0.3)** into a clean, standardized, fixed-length, labeled, patient-level-split dataset ready for downstream classification models.

---

## 1. Pipeline Architecture

```text
       ┌──────────────────────────────┐
       │   Raw PCG Audio Recording    │ (4000 Hz, 16-bit PCM Mono)
       └──────────────┬───────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │   Waveform & I/O Validation  │ (Check headers, corruption, channels)
       └──────────────┬───────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │   Polyphase Resampling       │ (4000 Hz ➔ 2000 Hz Target)
       └──────────────┬───────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │   Butterworth Bandpass (SOS) │ (25 Hz - 400 Hz, Order 5, Zero-phase)
       └──────────────┬───────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │   Optional Notch Filter      │ (50 Hz / 60 Hz Powerline Hum, Disabled by default)
       └──────────────┬───────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │  Spike & Motion Attenuation  │ (Energy thresholding)
       └──────────────┬───────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │   Peak Normalization         │ (Scale absolute peak to 0.95, Zero-safe)
       └──────────────┬───────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │   Fixed-Length Segmentation  │ (Window: 5.0s = 10,000 samples | Hop: 2.5s = 5,000 samples)
       └──────────────┬───────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │   Zero-Padding / Truncation  │ (Short recordings <5s & trailing edges padded to 10,000)
       └──────────────┬───────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │   Label Assignment           │ (0: Normal / Murmur Absent | 1: Murmur Present / Abnormal)
       └──────────────┬───────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │   Patient-Level Splitting    │ (70% Train, 15% Val, 15% Test | ZERO Patient Leakage)
       └──────────────┬───────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │  Training-Only Augmentation  │ (Time Shift, Gaussian Noise std=0.005, Time Stretch 0.95-1.05)
       └──────────────┬───────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │   Standardized HDF5 Output   │ (data/processed/module1_processed.h5)
       └──────────────────────────────┘
```

---

## 2. DSP & Preprocessing Specifications

| Component | Default Setting | Description |
| :--- | :--- | :--- |
| **Native Sampling Rate** | `4,000 Hz` | Uniform across all CirCor recordings |
| **Target Sampling Rate** | `2,000 Hz` | Standard downsampling via polyphase filtering (`scipy.signal.resample_poly`) |
| **Bandpass Filter** | `25 - 400 Hz` | Order 5 Butterworth zero-phase SOS filtering (`sosfiltfilt`), eliminating low-frequency baseline wander and high-frequency acoustic noise |
| **Notch Filter** | `Disabled` (50 Hz / 60 Hz configurable) | IIR notch filter with quality factor $Q=30$ |
| **Spike Attenuation** | `Enabled` | Clips mechanical stethoscope rubbing artifacts beyond $3.0 \times \text{std}$ |
| **Normalization** | `Peak Norm (0.95)` | $x_{\text{norm}} = \frac{x}{\max(|x|) + \epsilon} \times 0.95$; safely handles flatline/zero signals |
| **Window Duration** | `5.0 seconds` | Exactly `10,000` samples per window at 2,000 Hz |
| **Hop Duration** | `2.5 seconds` | 50% overlap (`5,000` samples step) |
| **Padding** | `Zero-padding` | Short recordings (<5.0s) and trailing windows are right-padded to exactly `10,000` samples |

---

## 3. Label Definition & Mapping

| Original `Murmur` Field | Clinical Interpretation | Module 1 Label | Cohort Assignment |
| :--- | :--- | :--- | :--- |
| **`Absent`** | No acoustic heart murmur detected | **`0` (Normal)** | Supervised Train / Val / Test (695 patients) |
| **`Present`** | Acoustic heart murmur definitively present | **`1` (Murmur / Abnormal)** | Supervised Train / Val / Test (179 patients) |
| **`Unknown`** | Indeterminate audio quality or diagnosis | **`-1` (Excluded)** | Excluded from standard supervised training (68 patients) |

---

## 4. Patient-Level Splitting & Zero-Leakage Guarantee

- **Partition Ratios**: **70% Train** (611 patients), **15% Validation** (131 patients), **15% Test** (132 patients).
- **Patient-Level Disjointness**: Partitioning is executed strictly on `Patient ID` before window expansion.
  - $\text{Train Patients} \cap \text{Validation Patients} = \emptyset$
  - $\text{Train Patients} \cap \text{Test Patients} = \emptyset$
  - $\text{Validation Patients} \cap \text{Test Patients} = \emptyset$
- **Multi-valve Preservation**: All recordings (`AV`, `MV`, `PV`, `TV`, `Phc`) for a given subject remain in that subject's assigned split.

---

## 5. Training-Only Data Augmentation

Augmentation is applied **strictly to training windows** (validation and test splits remain pristine and unaugmented):
1. **Time Shifting**: Circular roll up to $\pm 0.2$ seconds ($\pm 400$ samples).
2. **Additive Gaussian Noise**: Zero-mean Gaussian jitter with $\sigma = 0.005$.
3. **Time Stretching**: Rate perturbation within $[0.95, 1.05]$, resampled back to exactly `10,000` samples.

---

## 6. Output Dataset Schema (`module1_processed.h5`)

The dataset is serialized to HDF5 format with gzip compression:

```text
module1_processed.h5
├── train/
│   ├── audio               (N_train, 10000)  [float32]
│   ├── labels              (N_train,)        [int64]
│   ├── patient_ids         (N_train,)        [string]
│   ├── recording_ids       (N_train,)        [string]
│   ├── locations           (N_train,)        [string]
│   ├── window_indices      (N_train,)        [int64]
│   └── source_filepaths    (N_train,)        [string]
├── validation/
│   ├── audio               (N_val, 10000)    [float32]
│   ├── labels              (N_val,)          [int64]
│   ├── patient_ids         (N_val,)          [string]
│   ├── recording_ids       (N_val,)          [string]
│   ├── locations           (N_val,)          [string]
│   ├── window_indices      (N_val,)          [int64]
│   └── source_filepaths    (N_val,)          [string]
└── test/
    ├── audio               (N_test, 10000)   [float32]
    ├── labels              (N_test,)         [int64]
    ├── patient_ids         (N_test,)         [string]
    ├── recording_ids       (N_test,)         [string]
    ├── locations           (N_test,)         [string]
    ├── window_indices      (N_test,)         [int64]
    └── source_filepaths    (N_test,)         [string]
```

---

## 7. How to Run and Test

### Execute Pipeline on CirCor Dataset:
```bash
python scripts/run_module1_preprocessing.py
```

### Run Unit Tests:
```bash
pytest tests/
```
