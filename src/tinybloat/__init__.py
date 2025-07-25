"""
A module with extra functionality for tinygrad.
Documentation assumes you have some familiarity tinygrad already,
so if you do you can jump right in!
"""

from .complex_tensor import ComplexTensor
from .quantization import QTensor
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
	"safety_functions",
	"nn",
	"F",
	"type_utils",
	"testing",
	
	"ComplexTensor",
	"diag",
	"is_tinygrad_module",
	"module_on_device",
	"move_to_device",
	"cast_to_dtype",
	"nonzero",
	"cat",
	"cumprod",
	"assert_same_device",
	"chunk",
	"clamp",
	"stack",
	"outer",
	"limit_float_precision",
	"limit_sint_precision",
	"limit_uint_precision",
	"shard_model_",
	"cast_to_supported_and_move_",
	"QTensor"
]
