import tinygrad
from typing import Union
import numpy as np
import gguf
from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES, QK_K
from gguf.quants import quant_shape_to_byte_shape, quant_shape_from_byte_shape
from .common import hsplit

class QTensor:
	def __init__(self,
				value: Union[tinygrad.Tensor, np.ndarray, list, tuple],
				qtype,
				device = None):
		
		if device is None:
			device = tinygrad.Device.DEFAULT
		
		if not isinstance(qtype, GGMLQuantizationType):
			# Only GGUF/GGML types are supported for now
			raise NotImplementedError
		
		self._block_size, self._type_size = GGML_QUANT_SIZES[qtype]
		self._qtype = qtype
		
		if isinstance(value, np.ndarray):
			value = tinygrad.Tensor(value, device = device)
		elif not isinstance(value, tinygrad.Tensor):
			# TODO: examine dtype?
			value = tinygrad.Tensor(np.array(value), device = device)
		assert isinstance(value, tinygrad.Tensor)
		self._tg = value
	
	def dequantize(self):
		if self._qtype == GGMLQuantizationType.F32:
			return self._tg.bitcast(tinygrad.dtypes.float)
		elif self._qtype == GGMLQuantizationType.F16:
			return self._tg.bitcast(tinygrad.dtypes.half)
		
		else:
			# for GGUF types
			blocks = self._tg.bitcast(tinygrad.dtypes.uint8)
			shape = blocks.shape
			n_blocks = blocks.numel() // self._type_size
			blocks = blocks.reshape(n_blocks, self._type_size)
			
			# Now we actually have to do the dtype-specific parts
			if self._qtype == GGMLQuantizationType.Q6_K:
				ql, rest = hsplit(blocks, [QK_K // 2])
				qh, rest = hsplit(rest, [QK_K // 4])
				scales, d = hsplit(rest, [QK_K // 16])
				
				scales = scales.bitcast(tinygrad.dtypes.int8).cast(tinygrad.dtypes.float)
				d = d.bitcast(tinygrad.dtypes.half).cast(tinygrad.dtypes.float)
				d = (d * scales).reshape(n_blocks, QK_K // 16, 1)
				
				tocat = []
				for shift in [0, 4]:
					tocat.append(ql.reshape(n_blocks, -1, 1, 64) >> shift)
				ql = tinygrad.Tensor.cat(*tocat, dim = 2)
				ql = (ql & 0x0F).reshape(n_blocks, -1, 32)
				
				tocat = []
				for shift in [0, 2, 4, 6]:
					tocat.append(qh.reshape(n_blocks, -1, 1, 32) >> shift)
				qh = tinygrad.Tensor.cat(*tocat, dim = 2)
				qh = (qh & 0x03).reshape((n_blocks, -1, 32))
				q = (ql | (qh << 4)).cast(tinygrad.dtypes.int8) - 32
				q = q.reshape(n_blocks, QK_K // 16, -1).cast(tinygrad.dtypes.float)
				print("tg d, q shapes:", d.shape, q.shape)
				blocks = (d * q).reshape((n_blocks, QK_K))
			else:
				raise NotImplementedError(f"{_get_ggml_qtype_name(qtype)} dequantization not yet implemented")
			
			# sanity check present in original gguf code
			assert blocks.dtype == tinygrad.dtypes.float
			assert blocks.shape[-1] == self._block_size
			
			# reshape into proper tensor shape
			return blocks.reshape(*quant_shape_from_byte_shape(shape, self._qtype) )
		
	

def _get_ggml_qtype_name(qtype):
	for k, v in GGMLQuantizationType.__dict__.items():
		if v == qtype:
			return k
	raise ValueError(f"No GGML type with enum value {qtype}")
	
