"""
Unit and integration tests for Audio Spectrogram Transformer (AST) feature extraction branch.
"""
import pytest
import numpy as np
import torch
from pathlib import Path

from src.ast.config import ASTConfig
from src.ast.ast_preprocessor import ASTFeaturePreprocessor
from src.ast.ast_model_loader import ASTModelLoader
from src.ast.ast_embedding_extractor import ASTEmbeddingExtractor, extract_ast_embedding
from src.utils.config import RAW_DATA_DIR

def test_ast_config_defaults():
    """Verify default AST configuration parameters."""
    cfg = ASTConfig()
    assert cfg.target_sample_rate == 16000
    assert cfg.num_mel_bins == 128
    assert cfg.max_length == 1024
    assert cfg.embedding_dim == 768
    assert cfg.pooling_method == "mean"
    assert cfg.normalize_embedding is True

def test_ast_preprocessor_resampling_and_spectrogram():
    """Verify preprocessor converts 2000Hz 5-second audio into (1, 1024, 128) tensor."""
    preprocessor = ASTFeaturePreprocessor()
    
    # 5.0 seconds of audio at 2000 Hz = 10,000 samples
    synthetic_wave = np.sin(2 * np.pi * 50 * np.linspace(0, 5.0, 10000)).astype(np.float32)
    inputs = preprocessor.extract_ast_inputs(synthetic_wave, orig_sr=2000)
    
    assert isinstance(inputs, torch.Tensor)
    assert inputs.shape == (1, 1024, 128)
    assert not torch.isnan(inputs).any()
    assert not torch.isinf(inputs).any()

def test_ast_model_loader():
    """Verify AST model loads into eval mode with no gradients."""
    loader = ASTModelLoader()
    model = loader.load_model()
    
    assert model is not None
    assert not model.training
    # Check that parameters do not require gradients
    for p in model.parameters():
        assert not p.requires_grad

def test_ast_embedding_dimension_and_normalization():
    """Verify AST embedding has exact shape (768,) and unit norm."""
    extractor = ASTEmbeddingExtractor()
    synthetic_audio = np.random.randn(10000).astype(np.float32) * 0.1
    
    emb = extractor.extract_embedding(synthetic_audio, orig_sr=2000)
    
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (768,)
    assert not np.isnan(emb).any()
    assert not np.isinf(emb).any()
    
    # L2 norm must be approximately 1.0
    norm = np.linalg.norm(emb)
    assert abs(norm - 1.0) < 1e-4

def test_ast_deterministic_inference():
    """Verify repeated inference on identical audio produces bit-exact deterministic output."""
    extractor = ASTEmbeddingExtractor()
    synthetic_audio = np.random.randn(10000).astype(np.float32) * 0.05
    
    emb1 = extractor.extract_embedding(synthetic_audio, orig_sr=2000)
    emb2 = extractor.extract_embedding(synthetic_audio, orig_sr=2000)
    
    assert np.allclose(emb1, emb2, atol=1e-6)

def test_ast_real_circor_recording():
    """Test AST embedding extraction on an existing CirCor PCG recording."""
    wav_path = Path(RAW_DATA_DIR) / "training_data" / "40840_AV.wav"
    if not wav_path.exists():
        pytest.skip("CirCor test file 40840_AV.wav not found.")
        
    import wave
    with wave.open(str(wav_path), "rb") as wf:
        n_frames = wf.getnframes()
        sr = wf.getframerate()
        raw_bytes = wf.readframes(n_frames)
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
    emb = extract_ast_embedding(audio, orig_sr=sr)
    assert emb.shape == (768,)
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-4
