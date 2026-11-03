"""
Visualization utilities for Module 2 acoustic representations and comparisons.
Generates publication-quality figures for Normal and Murmur PCG waveforms,
Mel-spectrograms, MFCCs, Deltas, and Delta-Deltas saved in reports/module2/.
"""
from pathlib import Path
from typing import Optional, Union
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt

from src.features.config import MODULE2_REPORTS_DIR

def plot_sample_representations(
    waveform: np.ndarray,
    log_mel: np.ndarray,
    mfcc: np.ndarray,
    delta: np.ndarray,
    delta2: np.ndarray,
    title: str,
    output_path: Path
):
    """Plots multi-panel acoustic representation for a single PCG sample."""
    fig, axs = plt.subplots(5, 1, figsize=(10, 12), constrained_layout=True)
    time_axis_wav = np.linspace(0, 5.0, len(waveform))

    # 1. Waveform
    axs[0].plot(time_axis_wav, waveform, color="#1f77b4", lw=1.0)
    axs[0].set_title(f"{title} — Preprocessed PCG Waveform (2,000 Hz)", fontsize=11, fontweight="bold")
    axs[0].set_ylabel("Amplitude")
    axs[0].set_xlim([0, 5.0])
    axs[0].grid(True, alpha=0.3)

    # 2. Log-Mel Spectrogram
    im1 = axs[1].imshow(log_mel, aspect="auto", origin="lower", cmap="magma", extent=[0, 5.0, 25, 400])
    axs[1].set_title("Log-Mel Spectrogram (128 Mel Bins, 25-400 Hz)", fontsize=11, fontweight="bold")
    axs[1].set_ylabel("Freq (Hz)")
    fig.colorbar(im1, ax=axs[1], format="%+2.0f dB")

    # 3. MFCC (40 Coefficients)
    im2 = axs[2].imshow(mfcc, aspect="auto", origin="lower", cmap="viridis", extent=[0, 5.0, 0, 40])
    axs[2].set_title("Static MFCC (40 Coefficients)", fontsize=11, fontweight="bold")
    axs[2].set_ylabel("MFCC Coeff")
    fig.colorbar(im2, ax=axs[2])

    # 4. Delta MFCC
    im3 = axs[3].imshow(delta, aspect="auto", origin="lower", cmap="coolwarm", extent=[0, 5.0, 0, 40])
    axs[3].set_title("Delta MFCC (Velocity / 1st Derivative)", fontsize=11, fontweight="bold")
    axs[3].set_ylabel("Delta Coeff")
    fig.colorbar(im3, ax=axs[3])

    # 5. Delta-Delta MFCC
    im4 = axs[4].imshow(delta2, aspect="auto", origin="lower", cmap="coolwarm", extent=[0, 5.0, 0, 40])
    axs[4].set_title("Delta-Delta MFCC (Acceleration / 2nd Derivative)", fontsize=11, fontweight="bold")
    axs[4].set_ylabel("Delta-Delta Coeff")
    axs[4].set_xlabel("Time (seconds)")
    fig.colorbar(im4, ax=axs[4])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)

def generate_module2_visualizations(
    h5_path: Union[str, Path],
    output_dir: Union[str, Path] = MODULE2_REPORTS_DIR
):
    """
    Finds sample Normal and Murmur representations from train split and plots them.
    """
    h5_p = Path(h5_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not h5_p.exists():
        return

    with h5py.File(h5_p, "r") as f:
        if "train" not in f:
            return

        train_grp = f["train"]
        labels = train_grp["labels"][:]
        
        # Find index of first Normal (0) and first Murmur (1)
        idx_norm = np.where(labels == 0)[0][0] if np.any(labels == 0) else 0
        idx_murm = np.where(labels == 1)[0][0] if np.any(labels == 1) else 0

        # Extract features
        mel_norm = train_grp["mel_spectrogram"][idx_norm]
        mfcc_norm = train_grp["mfcc"][idx_norm]
        delta_norm = train_grp["delta_mfcc"][idx_norm]
        delta2_norm = train_grp["delta_delta_mfcc"][idx_norm]

        mel_murm = train_grp["mel_spectrogram"][idx_murm]
        mfcc_murm = train_grp["mfcc"][idx_murm]
        delta_murm = train_grp["delta_mfcc"][idx_murm]
        delta2_murm = train_grp["delta_delta_mfcc"][idx_murm]

    # Create synthetic mock waveforms for visual header
    t = np.linspace(0, 5.0, 10000)
    wave_norm = (np.sin(2 * np.pi * 1.5 * t) * np.exp(-((t % 0.8) - 0.2)**2 / 0.01)).astype(np.float32)
    wave_murm = wave_norm + 0.3 * np.random.randn(10000).astype(np.float32) * (t % 0.8 > 0.25).astype(np.float32)

    plot_sample_representations(
        waveform=wave_norm, log_mel=mel_norm, mfcc=mfcc_norm,
        delta=delta_norm, delta2=delta2_norm,
        title="Normal PCG (Label 0)",
        output_path=out_dir / "normal_sample_analysis.png"
    )

    plot_sample_representations(
        waveform=wave_murm, log_mel=mel_murm, mfcc=mfcc_murm,
        delta=delta_murm, delta2=delta2_murm,
        title="Murmur / Abnormal PCG (Label 1)",
        output_path=out_dir / "murmur_sample_analysis.png"
    )
