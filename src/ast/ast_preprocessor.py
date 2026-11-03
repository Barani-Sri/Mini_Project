"""
Audio Preprocessing and Spectrogram Conversion for Audio Spectrogram Transformer (AST).
Resamples heart sound signals to 16,000 Hz and extracts normalized Log-Mel spectrograms
of shape (batch_size, max_length, num_mel_bins) matching the AST vision-transformer input format.
"""
from typing import Optional, Tuple, Union
import numpy as np
import torch
import librosa
from transformers import ASTFeatureExtractor

from src.ast.config import ASTConfig
from src.audio.preprocessing import resample_pcg

class ASTFeaturePreprocessor:
    """
    Converts standardized PCG waveforms into AST-compatible Log-Mel Spectrogram inputs.
    """
    def __init__(self, config: Optional[ASTConfig] = None):
        self.config = config if config is not None else ASTConfig()
        try:
            self.feature_extractor = ASTFeatureExtractor.from_pretrained(
                self.config.model_name,
                sampling_rate=self.config.target_sample_rate,
                num_mel_bins=self.config.num_mel_bins,
                max_length=self.config.max_length,
                do_normalize=True
            )
        except Exception:
            # Fallback local initialization if offline/cache-only
            self.feature_extractor = ASTFeatureExtractor(
                feature_size=1,
                sampling_rate=self.config.target_sample_rate,
                num_mel_bins=self.config.num_mel_bins,
                max_length=self.config.max_length,
                do_normalize=True
            )

    def preprocess_waveform(
        self,
        waveform: np.ndarray,
        orig_sr: int = 2000
    ) -> np.ndarray:
        """
        Resamples input audio waveform to AST target rate (16,000 Hz).
        """
        if len(waveform) == 0:
            return np.zeros(self.config.target_sample_rate * 5, dtype=np.float32)
            
        # Resample from orig_sr (default 2000 Hz from Module 1) to target_sr (16,000 Hz)
        resampled, _ = resample_pcg(
            waveform,
            orig_sr=orig_sr,
            target_sr=self.config.target_sample_rate
        )
        return resampled.astype(np.float32)

    def extract_ast_inputs(
        self,
        waveform: np.ndarray,
        orig_sr: int = 2000
    ) -> torch.Tensor:
        """
        Transforms waveform into AST tensor inputs of shape (1, max_length, num_mel_bins).
        """
        # 1. Resample to 16 kHz
        resampled_audio = self.preprocess_waveform(waveform, orig_sr=orig_sr)
        
        # 2. Extract AST input features via ASTFeatureExtractor
        inputs = self.feature_extractor(
            resampled_audio,
            sampling_rate=self.config.target_sample_rate,
            return_tensors="pt"
        )
        
        # Tensor shape: (1, 1024, 128)
        input_values = inputs["input_values"]
        return input_values
