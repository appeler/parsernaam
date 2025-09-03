#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration constants for parsernaam.

This module contains all the hardcoded constants used throughout
the parsernaam package, including model parameters, file paths,
and classification categories.
"""

class ModelConfig:
    """Model configuration constants.
    
    Contains all the hyperparameters and settings used by the LSTM models
    for name parsing, including architecture parameters and file locations.
    
    Attributes:
        HIDDEN_SIZE: Dimension of LSTM hidden layers
        NUM_LAYERS: Number of LSTM layers in the model
        SEQUENCE_LENGTH: Maximum length of input name sequences
        CATEGORIES_SINGLE: Classification labels for single names
        CATEGORIES_POSITIONAL: Classification labels for multi-word names
        MODEL_FILES: Paths to model and vocabulary files
    """
    HIDDEN_SIZE = 256  # Dimension of LSTM hidden state
    NUM_LAYERS = 2     # Number of LSTM layers
    SEQUENCE_LENGTH = 30  # Maximum character sequence length
    
    # Classification categories for single names (first name only or last name only)
    CATEGORIES_SINGLE = ['last', 'first']
    
    # Classification categories for multi-word names (position-based)
    CATEGORIES_POSITIONAL = ['last_first', 'first_last']
    
    # File paths for trained models and vocabulary
    MODEL_FILES = {
        'single': "models/parsernaam.pt",      # Single name classifier
        'positional': "models/parsernaam_pos.pt",  # Positional classifier
        'vocab': "models/parsernaam.joblib"    # Character vocabulary
    }