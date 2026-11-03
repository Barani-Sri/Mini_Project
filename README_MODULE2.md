# Module 2: Heart Sound Feature Extraction & Multi-Location Representation

Module 2 transforms standardized, fixed-length PCG waveforms `(N, 10000)` produced by **Module 1** into deep-learning-ready acoustic representations, location-aware encodings, and multi-location patient tensors `(P, 5, 120, T)`.

---

## 1. System Architecture

```text
       ┌────────────────────────────────────────┐
       │   Module 1 Processed PCG Waveforms     │ (data/processed/module1_processed.h5)
       │         Waveform: (N, 10000)           │ (2000 Hz, 5.0 seconds)
       └──────────────────┬─────────────────────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
 ┌──────────────────────┐    ┌──────────────────────┐
 │ Log-Mel Spectrogram  │    │   40 Static MFCCs    │
 │    (128 Mel Bins)    │    │      (40, T)         │
 │     Shape: (128, T)  │    └──────────┬───────────┘
 └──────────────────────┘               │
                                        ▼
                             ┌──────────────────────┐
                             │ Delta MFCCs (40, T)  │
                             │ Delta2 MFCCs (40, T) │
                             └──────────┬───────────┘
                                        │
                                        ▼
                             ┌──────────────────────┐
                             │    Combined MFCC     │ (Concat along channel axis)
                             │     Shape: (120, T)  │
                             └──────────┬───────────┘
                                        │
                          ┌─────────────┴─────────────┐
                          ▼                           ▼
               ┌──────────────────────┐    ┌──────────────────────┐
               │  Quality Assessment  │    │  Location Encoding   │
               │ Score [0, 1] & Flags │    │ [AV:0, MV:1, PV:2,   │
               │  GOOD/QUEST/POOR     │    │  TV:3, Phc:4]        │
               └──────────────────────┘    └──────────┬───────────┘
                                                      │
                                                      ▼
                                           ┌──────────────────────┐
                                           │ Patient-Level Group  │
                                           │ Missing-Loc Padding  │
                                           │ Location Mask (5,)   │
                                           └──────────┬───────────┘
                                                      │
                                                      ▼
                                           ┌──────────────────────┐
                                           │ Fusion-Ready Tensors │
                                           │  Shape: (5, 120, T)  │
                                           └──────────┬───────────┘
                                                      │
                                                      ▼
                                           ┌──────────────────────┐
                                           │ Module 3 Classifiers │
                                           └──────────────────────┘
```

---

## 2. Feature Extraction Parameters

| Component | Default Value | Description |
| :--- | :--- | :--- |
| **Sampling Rate ($f_s$)** | `2,000 Hz` | Standardized rate from Module 1 |
| **Frame Length** | `30.0 ms` (`60` samples) | Short-time Fourier transform window length |
| **Hop Length** | `10.0 ms` (`20` samples) | 66.7% frame overlap |
| **FFT Size ($N_{\text{fft}}$)** | `512` | Zero-padded frequency resolution ($3.9$ Hz/bin) |
| **Frequency Range** | `25 Hz - 400 Hz` | Filtered PCG diagnostic bandwidth |
| **Log-Mel Bands** | `128` bands | Log-scaled power mel-spectrogram with per-window standardization |
| **MFCC Coefficients** | `40` static coefficients | Discrete Cosine Transform cepstral coefficients |
| **Delta Features** | `40` velocity + `40` acceleration | 1st and 2nd temporal derivatives |
| **Combined MFCC Channels** | `120` channels | Static ($40$) + Delta ($40$) + Delta-Delta ($40$) |
| **Time Dimension ($T$)** | `501` frames | $1 + \lfloor 10000 / 20 \rfloor = 501$ |

---

## 3. Location Encoding & Multi-Location Patient Grouping

PCG auscultations are deterministically mapped to 5 cardiac valve areas:

$$0 \rightarrow \mathbf{AV},\quad 1 \rightarrow \mathbf{MV},\quad 2 \rightarrow \mathbf{PV},\quad 3 \rightarrow \mathbf{TV},\quad 4 \rightarrow \mathbf{Phc}$$

### Missing Location Handling & Masks
For patients missing one or more recordings:
1. Missing locations are **zero-padded** in the feature representation: $\mathbf{0} \in \mathbb{R}^{120 \times T}$.
2. A binary **`location_mask`** $\in \{0, 1\}^5$ indicates available valve sites (e.g. $[1, 1, 0, 1, 0]$).
3. The resulting patient tensor has unified shape:

$$\text{Patient Tensor} \in \mathbb{R}^{5 \times 120 \times T}$$

---

## 4. Multi-Location Fusion Modes

Module 2 provides three standardized downstream representation formats:

1. **Early Fusion**: Concatenates/flattens location representations along channel dimension: $(5 \times 120, T) = (600, T)$.
2. **Intermediate Fusion**: Retains explicit $(5, 120, T)$ multi-valve tensor for multi-branch neural encoders.
3. **Attention-Ready Fusion**: Supplies paired location features $(B, 5, 120, T)$ and location masks $(B, 5)$ for cross-valve attention mechanisms.

---

## 5. Feature Quality Assessment

Each window is evaluated across 7 signal metrics:
* RMS energy & standard deviation
* Near-zero sample fraction ($|x| < 10^{-4}$)
* Signal clipping fraction ($|x| \ge 0.949$)
* Non-finite checking ($\text{NaN} / \text{Inf}$)
* Numerical MFCC & Mel spectrogram validity

Quality Score $\in [0, 1]$ & Discrete Flags:
* **`GOOD`**: $\text{score} \ge 0.70$
* **`QUESTIONABLE`**: $0.30 \le \text{score} < 0.70$
* **`POOR`**: $\text{score} < 0.30$

*Note: Recordings are never discarded based on quality score; metadata flags are preserved for downstream confidence calibration.*

---

## 6. Output Dataset Schema (`module2_features.h5`)

```text
module2_features.h5
├── train/
│   ├── mfcc                (N_train, 40, 501)    [float32]
│   ├── delta_mfcc          (N_train, 40, 501)    [float32]
│   ├── delta_delta_mfcc    (N_train, 40, 501)    [float32]
│   ├── mfcc_combined       (N_train, 120, 501)   [float32]
│   ├── mel_spectrogram     (N_train, 128, 501)   [float32]
│   ├── labels              (N_train,)            [int64]
│   ├── location_ids        (N_train,)            [int64]
│   ├── patient_ids         (N_train,)            [string]
│   ├── recording_ids       (N_train,)            [string]
│   ├── locations           (N_train,)            [string]
│   ├── quality_scores      (N_train,)            [float32]
│   └── quality_flags       (N_train,)            [string]
├── validation/
│   └── ... (same structure for validation split)
├── test/
│   └── ... (same structure for test split)
└── patients/
    ├── train/
    │   ├── patient_features (P_train, 5, 120, 501) [float32]
    │   ├── location_masks   (P_train, 5)           [float32]
    │   ├── patient_labels   (P_train,)             [int64]
    │   └── patient_ids      (P_train,)             [string]
    ├── validation/
    │   └── ... (P_val, 5, 120, 501)
    └── test/
        └── ... (P_test, 5, 120, 501)
```

---

## 7. How to Execute & Test

### Run Feature Extraction on Real Dataset:
```bash
python scripts/run_module2_features.py
```

### Run All Unit Tests:
```bash
pytest tests/
```
