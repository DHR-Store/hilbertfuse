"""hilbertfuse - MI gene selection + 2D gene images + pretrained CNN + Guided Grad-CAM."""
from .pipeline import PanCancerClassifier, EnsembleClassifier, cross_validate, preset
from .hilbertfusenet import HilbertFuseNet
from .hybrid import HybridClassifier
from .preprocess import GeneImagePreprocessor
from .data import load_table
from .synthetic import make_synthetic

__all__ = ["PanCancerClassifier", "EnsembleClassifier", "HybridClassifier", "HilbertFuseNet", "preset", "cross_validate", "GeneImagePreprocessor",
           "load_table", "make_synthetic"]
__version__ = "1.0.0"
