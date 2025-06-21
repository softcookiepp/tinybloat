import tinygrad
from tinygrad.device import is_dtype_supported
from .complex_tensor import ComplexTensor
from typing import Union

TensorTypes = Union[tinygrad.Tensor, ComplexTensor]

FLOAT_TYPES = [
	tinygrad.dtypes.fp8e4m3,
	tinygrad.dtypes.fp8e5m2,
	tinygrad.dtypes.bfloat16,
	tinygrad.dtypes.half,
	tinygrad.dtypes.float,
	tinygrad.dtypes.double
]

FLOAT_TYPES = sorted(FLOAT_TYPES, key = lambda x: x.itemsize)

def is_floating_point(t):
	return t.dtype in FLOAT_TYPES

def is_uint(t):
	raise NotImplementedError
