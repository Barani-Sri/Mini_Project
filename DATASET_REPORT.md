# DATASET REPORT: CirCor DigiScope Phonocardiogram Dataset (v1.0.3)

> **Dataset Name**: The CirCor DigiScope Phonocardiogram Dataset  
> **Source**: PhysioNet (George B. Moody PhysioNet Challenge 2022)  
> **Official URL**: `https://physionet.org/content/circor-heart-sound/1.0.3/`  
> **Storage Location**: `data/raw/circor/`  
> **Verification Status**: ✅ Complete & Integrity Verified (0 corrupted files)  
> **Inspection Date**: 2026-08-22  

---

## 1. Executive Summary & Verification

The **CirCor DigiScope Phonocardiogram Dataset** was downloaded directly from official PhysioNet servers and verified locally. All archives, audio files, metadata spreadsheets, header descriptors, and cycle segmentations are fully extracted and structurally valid.

| Metric | Value |
| :--- | :--- |
| **Total Enrolled Patients** | **942** subjects |
| **Total Audio Recordings (.wav)** | **3,163** PCG audio files |
| **Total Segment Annotations (.tsv)** | **3,163** segmentation files |
| **Total Header Files (.hea)** | **3,163** WFDB header files |
| **Total Extracted Dataset Files** | **10,435** files |
| **Corrupted / Invalid Audio Files** | **0** (100% readable) |
| **Total Audio Duration** | **20.09 hours** (72,338.8 seconds) |
| **Primary Audio Sample Rate** | **4,000 Hz** (100% of recordings) |
| **Audio Format** | **16-bit PCM Mono WAV** |

---

## 2. Dataset Directory Structure

```text
data/
└── raw/
    ├── circor/
    │   ├── training_data.csv               # Master patient demographic, clinical & murmur annotations (942 rows)
    │   ├── RECORDS                         # List of patient record identifiers
    │   ├── LICENSE.txt                     # Open Access / PhysioNet License
    │   ├── SHA256SUMS.txt                  # Official cryptographic checksums
    │   └── training_data/                  # Directory containing all audio and annotation files
    │       ├── <PatientID>.txt             # Individual clinical report
    │       ├── <PatientID>.hea             # Patient master header
    │       ├── <PatientID>_<Location>.wav  # Raw PCG audio recording (4000 Hz, 16-bit PCM)
    │       ├── <PatientID>_<Location>.hea  # Individual recording header
    │       └── <PatientID>_<Location>.tsv  # Four-state PCG segmentation boundaries
    └── zips/
        └── the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip (449.45 MB)
```

---

## 3. Audio Recording Specifications

All audio files in the dataset have been opened, decoded, and analyzed:

### 3.1 Audio Parameters
* **Sampling Rate**: `4,000 Hz` (Uniform across all 3,163 files)
* **Channels**: `1` (Single-channel Mono audio)
* **Bit Depth**: `16-bit` linear PCM
* **Encoding**: Standard WAV (RIFF/WAVE header)
* **Audio Quality**: High signal fidelity captured via DigiScope electronic stethoscope

### 3.2 Duration Distribution
* **Minimum Duration**: `5.15 s`
* **Maximum Duration**: `64.51 s`
* **Mean Duration**: `22.87 s` (± `7.28 s`)
* **Median Duration**: `21.46 s`
* **Interquartile Range**: `19.06 s` (25th) to `29.39 s` (75th percentile)

### 3.3 Auscultation Location Breakdown
PCG audio is recorded across four standard cardiac auscultation valve sites (and occasional pulmonary/precordial checkpoints):

| Auscultation Site Code | Cardiac Valve / Location | Recording Count | Percentage |
| :--- | :--- | :--- | :--- |
| **AV** | Aortic Valve Area (2nd Right Intercostal Space) | 787 | 24.9% |
| **MV** | Mitral Valve Area (Cardiac Apex) | 849 | 26.8% |
| **PV** | Pulmonary Valve Area (2nd Left Intercostal Space) | 762 | 24.1% |
| **TV** | Tricuspid Valve Area (Lower Left Sternal Border) | 720 | 22.8% |
| **Phc** | Precordium / Other Checkpoint | 2 | 0.1% |

---

## 4. Patient Demographics & Clinical Metadata

The master table `training_data.csv` contains 942 patients and 23 clinical attributes:

### 4.1 Age Demographics
```text
Age
Child         664
Infant        126
NaN            74
Adolescent     72
Neonate         6
```

### 4.2 Sex Demographics
```text
Sex
Female    486
Male      456
```

### 4.3 Height & Weight
* **Height (cm)**: Mean `110.8 cm` (Range: `35.0` - `180.0` cm)
* **Weight (kg)**: Mean `23.6 kg` (Range: `2.3` - `110.8` kg)
* **Pregnancy Status**: 70 pregnant, 872 non-pregnant

---

## 5. Murmur & Outcome Annotations

### 5.1 Murmur Status Breakdown (Patient Level)
* **Absent**: **695** subjects (73.8%)
* **Present**: **179** subjects (19.0%)
* **Unknown / Unannotated**: **68** subjects (7.2%)

### 5.2 Clinical Outcome Breakdown (Patient Level)
* **Normal Outcome**: **486** subjects (51.6%)
* **Abnormal Outcome**: **456** subjects (48.4%)

### 5.3 Cross-Tabulation: Murmur vs. Clinical Outcome
```text
Outcome  Abnormal  Normal  All
Murmur                        
Absent        263     432  695
Present       150      29  179
Unknown        43      25   68
All           456     486  942
```

### 5.4 Fine-Grained Murmur Morphological Features
For subjects with `Murmur == Present` (179 subjects):
* **Timing**: Holosystolic, Early-systolic, Mid-systolic, Late-systolic, Continuous
* **Shape**: Diamond (Crescendo-Decrescendo), Plateau, Decrescendo, Crescendo
* **Grading**: Grade I/VI to Grade IV/VI
* **Pitch**: High, Medium, Low
* **Quality**: Harsh, Blowing, Musical

---

## 6. PCG Fundamental Heart Sound Segmentation (.tsv)

The dataset contains Springer/Schmidt segmentation annotations dividing PCG cycles into 4 distinct physiological states:

1. **State 1 — S1**: First heart sound (closure of Mitral/Tricuspid valves, marking start of ventricular systole).
2. **State 2 — Systole**: Systolic ejection period between S1 and S2 (where systolic murmurs reside).
3. **State 3 — S2**: Second heart sound (closure of Aortic/Pulmonary valves, marking start of ventricular diastole).
4. **State 4 — Diastole**: Diastolic ventricular filling period between S2 and S1.

Sample segment duration statistics:
* **S1 Mean Duration**: `115.3 ms`
* **Systole Mean Duration**: `140.5 ms`
* **S2 Mean Duration**: `98.7 ms`
* **Diastole Mean Duration**: `244.2 ms`

---

## 7. Binary Label Mapping Strategy for Modeling

For binary classification (`Normal` vs `Murmur / Abnormal`), two rigorous clinical formulation strategies are established:

### Strategy 1: Murmur Detection Binary Task (Primary Acoustic Task)
* **Class 0 (Normal / Murmur Absent)**: `Murmur == 'Absent'` (695 patients)
* **Class 1 (Murmur Present / Abnormal Acoustic)**: `Murmur == 'Present'` (179 patients)
* **Ambiguous Cases**: `Murmur == 'Unknown'` (68 patients) — excluded from standard training/evaluation or treated as separate confidence cohort.
* **Imbalance Ratio**: **3.88 : 1** (Negative:Positive).

### Strategy 2: Clinical Outcome Binary Task (Holistic Diagnostic Task)
* **Class 0 (Normal)**: `Outcome == 'Normal'` (486 patients)
* **Class 1 (Abnormal)**: `Outcome == 'Abnormal'` (456 patients)
* **Imbalance Ratio**: **1.07 : 1** (Negative:Positive).

### Patient-Level vs. Recording-Level Multi-Instance Formulation
* Patient diagnosis incorporates all available recordings from all four auscultation valves (`AV`, `PV`, `TV`, `MV`).
* Recording-level models evaluate segment auscultation windows, while patient-level aggregator aggregates recording probabilities via max-pooling / attention pooling.

---

## 8. Verification Checklist

- [x] Official PhysioNet CirCor dataset v1.0.3 downloaded directly from official archive.
- [x] Archive checksum and zip structure integrity verified.
- [x] Dataset extracted to canonical path `data/raw/circor/`.
- [x] All 3,163 PCG WAV recordings decoded without any corruption.
- [x] Audio properties verified: 4,000 Hz, 16-bit PCM, Mono.
- [x] Demographic, clinical, murmur, and outcome metadata verified.
- [x] Segment annotations (.tsv) and headers (.hea) verified.
- [x] Binary label formulation validated for machine learning pipeline.
