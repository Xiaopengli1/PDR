"""
PDR Core Library - Personalized Deep Research framework.
Suppresses Transformers advisory warnings for cleaner output.
"""
import os

# Suppress Transformers advisory warnings
# None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
