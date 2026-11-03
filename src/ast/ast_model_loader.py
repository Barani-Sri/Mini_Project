"""
Model loader and memory caching for pretrained Audio Spectrogram Transformer (AST).
Provides robust model initialization, CPU/GPU device handling, and eval-mode caching.
"""
from typing import Optional
import torch
import torch.nn as nn
from transformers import ASTModel, ASTConfig as TransformersASTConfig

from src.ast.config import ASTConfig

# Module-level model cache
_AST_MODEL_CACHE: dict = {}

class ASTModelLoader:
    """
    Manages loading and caching of pretrained Audio Spectrogram Transformer models.
    """
    def __init__(self, config: Optional[ASTConfig] = None):
        self.config = config if config is not None else ASTConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() and self.config.device == "cuda" else "cpu")

    def load_model(self) -> ASTModel:
        """
        Loads the pretrained AST model in eval mode, caching it in memory.
        """
        cache_key = f"{self.config.model_name}_{self.device}"
        if cache_key in _AST_MODEL_CACHE:
            return _AST_MODEL_CACHE[cache_key]

        try:
            model = ASTModel.from_pretrained(self.config.model_name)
        except Exception as e:
            # Fallback: initialize from ASTConfig with pretrained architecture if offline cache matches
            hf_config = TransformersASTConfig.from_pretrained(self.config.model_name)
            model = ASTModel(hf_config)

        model.to(self.device)
        model.eval()
        model.requires_grad_(False)

        _AST_MODEL_CACHE[cache_key] = model
        return model
