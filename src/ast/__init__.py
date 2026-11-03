"""
Audio Spectrogram Transformer (AST) Representation Branch.
Provides pretrained vision-transformer feature extraction for heart sound phonocardiograms.
"""
from src.ast.config import ASTConfig, DEFAULT_AST_MODEL_NAME
from src.ast.ast_preprocessor import ASTFeaturePreprocessor
from src.ast.ast_model_loader import ASTModelLoader
from src.ast.ast_embedding_extractor import ASTEmbeddingExtractor, extract_ast_embedding

__all__ = [
    "ASTConfig",
    "DEFAULT_AST_MODEL_NAME",
    "ASTFeaturePreprocessor",
    "ASTModelLoader",
    "ASTEmbeddingExtractor",
    "extract_ast_embedding"
]
