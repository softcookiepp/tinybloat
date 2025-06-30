import tinygrad
from typing import Union
import numpy as np


quantization_methods = [
	"q4_0",
	"q4_1",
	"q8_0",
	"q8_1"
]


def quantize_q4_0(value: tinygrad.Tensor, block_size: int):
	assert block_size > 0
	
	# reshape thingy to thingy
	value = value.reshape(-1, block_size)
	# actually
	(value.abs().max(1) / 7)
	
	raise NotImplementedError
	
def dequantize_q4_0(value: tinygrad.Tensor, block_size: int):
	assert value.dtype == tinygrad.dtypes.int8
	raise NotImplementedError
	
def pack_to_int4(int8_tensor: tinygrad.Tensor):
	"""
	Converts a int8 tensor into a int4 tensor
	"""
	raise NotImplementedError
	
	# first we gotta ensure the new shape is all even numbers
	underlying_shape = []
	shape = []
	for d in int8_tensor.shape:
		if d % 2 > 0:
			d += 1
		shape.append(d)
		underlying_shape.append(d // 2)
	

class QuantizedTensor:
	def __init__(self,
				value: Union[tinygrad.Tensor, np.ndarray, list, tuple],
				quantization_method: str,
				block_size = None,
				device = None):
		raise NotImplementedError
		assert quantization_method in quantization_methods

	
