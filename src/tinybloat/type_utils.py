import tinygrad
from tinygrad.device import is_dtype_supported
from .complex_tensor import ComplexTensor
from typing import Union

TensorTypes = Union[tinygrad.Tensor, ComplexTensor]
