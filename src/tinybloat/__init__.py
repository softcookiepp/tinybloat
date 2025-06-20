"""
A module with extra functionality for tinygrad.
Documentation assumes you have some familiarity tinygrad already,
so if you do you can jump right in!
"""

from .complex_tensor import ComplexTensor
from . import linalg
from .common import *
from .safety_functions import *
from .memory import *
from . import nn
from . import F
from .compatibility import *
from . import type_utils

__all__ = [
	"linalg",
	"ComplexTensor"
]
