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
	# first we gotta ensure the new shape is all even numbers
	int8_tensor = int8_tensor.reshape(-1)
	if int8_tensor.shape[0] % 2 > 0:
		int8_tensor = int8_tensor.pad([0, 1])
	int8_tensor = int8_tensor.reshape(-1, 2)
	
	# Ok, now it should be reshaped into groups of 2.
	# We can now do the thingy where we convert the bits!
	a = int8_tensor[:, 0]
	b = int8_tensor[:, 1]
	
	out = (a << 4) | (b & 0x0F)
	print(out.numpy() )
	raise NotImplementedError

class QuantizedTensor:
	def __init__(self,
				value: Union[tinygrad.Tensor, np.ndarray, list, tuple],
				quantization_method: str,
				block_size = None,
				device = None):
		raise NotImplementedError
		assert quantization_method in quantization_methods

	
