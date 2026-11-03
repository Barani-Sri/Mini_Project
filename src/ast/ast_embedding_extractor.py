"""
AST Embedding Extraction Engine.
Generates fixed-dimensional (768-dim), deterministic, L2-normalized AST acoustic embeddings
from heart sound PCG audio waveforms.
"""
from typing import Optional, Union
import numpy as np
import torch
import torch.nn.functional as F

from src.ast.config import ASTConfig
from src.ast.ast_preprocessor import ASTFeaturePreprocessor
from src.ast.ast_model_loader import ASTModelLoader

class ASTEmbeddingExtractor:
    """
    Extracts fixed-dimensional representations from heart sound PCG signals using AST.
    """
    def __init__(self, config: Optional[ASTConfig] = None):
        self.config = config if config is not None else ASTConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() and self.config.device == "cuda" else "cpu")
        self.preprocessor = ASTFeaturePreprocessor(self.config)
        self.loader = ASTModelLoader(self.config)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = self.loader.load_model()
        return self._model

    def extract_embedding(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
        orig_sr: int = 2000
    ) -> np.ndarray:
        """
        Extracts a fixed 768-dimensional AST embedding for a heart sound recording.
        
        Args:
            waveform: 1D audio array (e.g. 5.0s @ 2000Hz from Module 1, or raw PCG).
            orig_sr: Sampling rate of input waveform (default 2000 Hz).
            
        Returns:
            embedding: 1D np.ndarray of shape (768,), float32, L2-normalized.
        """
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu().numpy()

        waveform = np.asarray(waveform, dtype=np.float32).ravel()

        if len(waveform) == 0:
            zeros = np.zeros(self.config.embedding_dim, dtype=np.float32)
            return zeros

        # 1. Preprocess & extract AST input spectrogram (1, 1024, 128)
        input_values = self.preprocessor.extract_ast_inputs(waveform, orig_sr=orig_sr)
        input_values = input_values.to(self.device)

        # 2. Forward pass in evaluation mode (no gradients, deterministic)
        with torch.no_grad():
            outputs = self.model(input_values=input_values)
            last_hidden_state = outputs.last_hidden_state # (1, sequence_length, 768)

            # 3. Pooling strategy
            if self.config.pooling_method == "cls":
                # First token is CLS
                embedding = last_hidden_state[:, 0, :] # (1, 768)
            elif self.config.pooling_method == "pooled" and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embedding = outputs.pooler_output # (1, 768)
            else: # "mean" (default)
                # Mean pooling across patch tokens (excluding CLS & Distillation tokens at indices 0 and 1)
                if last_hidden_state.shape[1] > 2:
                    patch_tokens = last_hidden_state[:, 2:, :]
                    embedding = patch_tokens.mean(dim=1) # (1, 768)
                else:
                    embedding = last_hidden_state.mean(dim=1)

            # 4. Optional L2 Unit Normalization
            if self.config.normalize_embedding:
                embedding = F.normalize(embedding, p=2, dim=-1, eps=1e-8)

        embedding_np = embedding.squeeze(0).cpu().numpy().astype(np.float32)
        return np.nan_to_num(embedding_np, nan=0.0, posinf=0.0, neginf=0.0)

    def __call__(self, waveform: Union[np.ndarray, torch.Tensor], orig_sr: int = 2000) -> np.ndarray:
        return self.extract_embedding(waveform, orig_sr=orig_sr)


def extract_ast_embedding(
    waveform: Union[np.ndarray, torch.Tensor],
    orig_sr: int = 2000,
    config: Optional[ASTConfig] = None
) -> np.ndarray:
    """
    Convenience function: extracts a 768-dimensional AST embedding from audio waveform.
    """
    extractor = ASTEmbeddingExtractor(config=config)
    return extractor.extract_embedding(waveform, orig_sr=orig_sr)
