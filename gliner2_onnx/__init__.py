"""GLiNER2 ONNX Runtime - NER and classification without PyTorch."""

from importlib.metadata import PackageNotFoundError, version

from .constants import Precision
from .exceptions import ConfigurationError, GLiNER2Error, ModelNotFoundError
from .runtime import GLiNER2ONNXRuntime
from .schema import Schema
from .types import Entity, ExtractionResult

try:
    __version__ = version("gliner2-onnx")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = [
    "ConfigurationError",
    "Entity",
    "ExtractionResult",
    "GLiNER2Error",
    "GLiNER2ONNXRuntime",
    "ModelNotFoundError",
    "Precision",
    "Schema",
    "__version__",
]
