import tinygrad
from tinygrad import Tensor, dtypes
from typing import Union
import numpy as np
import gguf
from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES, QK_K
from gguf.quants import quant_shape_to_byte_shape, quant_shape_from_byte_shape
from .common import hsplit, broadcast_lshift, broadcast_rshift
from .compatibility import device_supports_dtype, convert_fp16, convert_bfloat16, convert_fp8e4m3, convert_fp8e5m2


def _get_scale_min(scales: tinygrad.Tensor):
	n_blocks = scales.shape[0]
	scales = scales.bitcast(dtypes.uint8)
	### Unpacking the following: ###
	#  0 EEAAAAAA
	#  1 FFBBBBBB
	#  2 GGCCCCCC
	#  3 HHDDDDDD
	#  4 eeaaaaaa
	#  5 ffbbbbbb
	#  6 ggcccccc
	#  7 hhdddddd
	#  8 eeeeEEEE
	#  9 ffffFFFF
	# 10 ggggGGGG
	# 11 hhhhHHHH
	scales = scales.reshape(n_blocks, 3, 4)
	d, m, m_d = scales.chunk(3, dim = -2)

	sc = Tensor.cat(*[d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim = -1)
	min = Tensor.cat(*[m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim = -1)

	return (sc.reshape(n_blocks, 8), min.reshape(n_blocks, 8))

def _convert_f16_to_f32(t):
	if device_supports_dtype(t.device, dtypes.half):
		return t.bitcast(dtypes.half).cast(dtypes.float)
	else:
		return QTensor(t, GGMLQuantizationType.F16).dequantize()

class QTensor:
	"""
	A tensor with support for quantized data types, particularly those from GGUF.
	It will also support lower-precision data types on hardware without native support
	in the future.
	"""
	def __init__(self,
				value: Union[tinygrad.Tensor, np.ndarray, list, tuple, gguf.gguf_reader.ReaderTensor],
				qtype,
				device = None,
				requires_grad = None,
				value_quantized = True
				):
		
		self._dequantized = None
		
		if device is None:
			if isinstance(value, Tensor):
				device = value.device
			else:
				device = tinygrad.Device.DEFAULT
		
		if isinstance(value, np.ndarray):
			value = tinygrad.Tensor(value, device = device, requires_grad = requires_grad)
		elif isinstance(value, gguf.gguf_reader.ReaderTensor):
			qtype = value.tensor_type
			value = tinygrad.Tensor(np.array(value.data), device = device, requires_grad = requires_grad)
			
		elif not isinstance(value, tinygrad.Tensor):
			# TODO: examine dtype?
			value = tinygrad.Tensor(np.array(value), device = device, requires_grad = requires_grad)
		assert isinstance(value, tinygrad.Tensor)
		
		if qtype is None:
			# just set it as dtype
			qtype = value.dtype
		self._qtype = qtype
		self._init_shape = None
		if hasattr(value, "shape"):
			self._init_shape = value.shape
		if isinstance(qtype, GGMLQuantizationType):
			self._block_size, self._type_size = GGML_QUANT_SIZES[qtype]
		elif isinstance(qtype, tinygrad.dtype.DType):
			if not device_supports_dtype(device, qtype):
				# TODO: implement software-level dequantization of regular tinygrad types
				value = value.bitcast(dtypes.uint8)
			else:
				# just set dequantized as
				if value_quantized:
					self._dequantized = value.bitcast(qtype)
				else:
					raise NotImplementedError
		else:
			# Only GGUF/GGML types and tinygrad dtypes are supported for now
			raise NotImplementedError
		self._tg = value
		
	def to(self, device: str):
		# we don't want to dequantize before moving to a different device.
		# so we must lay this over
		raise NotImplemented
	
	def dequantize(self):
		"""
		Returns the dequantized tensor, unrealized.
		"""
		# just to make things easier...
		if not self._dequantized is None:
			return self._dequantized	
		
		elif self._qtype == GGMLQuantizationType.F32 or self._qtype == dtypes.float:
			self._dequantized = self._tg.bitcast(dtypes.float)
		elif self._qtype == GGMLQuantizationType.F16 or self._qtype == dtypes.half:
			if device_supports_dtype(self._tg.device, dtypes.half):
				self._dequantized = self._tg.bitcast(dtypes.half)
			else:
				self._dequantized = convert_fp16(self._tg, dtypes.float)
		
		elif self._qtype == GGMLQuantizationType.BF16 or self._qtype == dtypes.bfloat16:
			self._dequantized = convert_bfloat16(self._tg, dtypes.float)
		
		elif self._qtype == dtypes.fp8e4m3:
			self._dequantized = convert_fp8e4m3(self._tg, dtypes.float)
		elif self._qtype == dtypes.fp8e5m2:
			self._dequantized = convert_fp8e5m2(self._tg, dtypes.float)
		elif self._qtype == dtypes.long:
			raise NotImplementedError
			self._dequantized = self._tg.bitcast(dtypes.int).reshape(-1, 2)[:, 0].reshape(*self._init_shape)
		elif self._qtype == dtypes.ulong:
			raise NotImplementedError
			self._dequantized = self._tg.bitcast(dtypes.uint).reshape(-1, 2)[:, 0].reshape(*self._init_shape)
		
		
		elif isinstance(self._qtype, GGMLQuantizationType):
			# for GGUF types
			blocks = self._tg.bitcast(dtypes.uint8)
			shape = blocks.shape
			n_blocks = blocks.numel() // self._type_size
			blocks = blocks.reshape(n_blocks, self._type_size)
			
			# Now we actually have to do the dtype-specific parts
			if self._qtype == GGMLQuantizationType.Q2_K:
				scales, rest = hsplit(blocks, [QK_K // 16])
				qs, rest = hsplit(rest, [QK_K // 4])
				d, dmin = hsplit(rest, [2])
				
				# this might work!
				d = _convert_f16_to_f32(d)
				dmin = _convert_f16_to_f32(dmin)

				# (n_blocks, 16, 1)
				dl = (d * (scales & 0xF).cast(dtypes.float32)).reshape(n_blocks, QK_K // 16, 1)
				ml = (dmin * (scales >> 4).cast(dtypes.float32)).reshape(n_blocks, QK_K // 16, 1)
				
				to_cat = []
				for shift in [0, 2, 4, 6]:
					to_cat.append( (qs.reshape(n_blocks, -1, 1, 32) >> shift) & 0x3 )
				qs = Tensor.cat(*to_cat, dim = 2)
				#qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & np.uint8(3)

				qs = qs.reshape(n_blocks, QK_K // 16, 16).cast(dtypes.float32)

				qs = dl * qs - ml

				blocks = qs.reshape((n_blocks, -1))
			
			elif self._qtype == GGMLQuantizationType.Q3_K:
				hmask, rest = hsplit(blocks, [QK_K // 8])
				qs, rest = hsplit(rest, [QK_K // 4])
				scales, d = hsplit(rest, [12])

				d = _convert_f16_to_f32(d)

				# The scales are packed at 6-bit each in this pattern:
				#  0: IIIIAAAA
				#  1: JJJJBBBB
				#  2: KKKKCCCC
				#  3: LLLLDDDD
				#  4: MMMMEEEE
				#  5: NNNNFFFF
				#  6: OOOOGGGG
				#  7: PPPPHHHH
				#  8: MMIIEEAA
				#  9: NNJJFFBB
				# 10: OOKKGGCC
				# 11: PPLLHHDD
				lscales, hscales = hsplit(scales, [8])
				lscales = lscales.reshape(n_blocks, 1, 8)
				lscales = (lscales >> 0).cat(lscales >> 4, dim = 1)
				lscales = lscales.reshape(n_blocks, 16)
				
				hscales = broadcast_rshift(hscales.reshape(n_blocks, 1, 4), [0, 2, 4, 6], 1)
				
				hscales = hscales.reshape(n_blocks, 16)
				scales = (lscales & 0x0F) | ( (hscales & 0x03) << 4 )
				scales = (scales.cast(dtypes.int8) - 32).cast(dtypes.float32)

				dl = (d * scales).reshape(n_blocks, 16, 1)
				
				ql = broadcast_rshift(qs.reshape(n_blocks, -1, 1, 32), [0, 2, 4, 6], 2)
				
				# and this one
				#qh = hmask.reshape(n_blocks, -1, 1, 32) >> np.array([i for i in range(8)], dtype=np.uint8).reshape((1, 1, 8, 1))
				qh = broadcast_rshift(hmask.reshape(n_blocks, -1, 1, 32), np.arange(8), 2)
				
				ql = ql.reshape(n_blocks, 16, QK_K // 16) & 0x3
				qh = qh.reshape(n_blocks, 16, QK_K // 16) & 0x1
				qh = qh ^ 0x1  # strangely, the offset is zero when the bitmask is 1
				q = (ql.cast(dtypes.int8) - (qh << 2).cast(dtypes.int8)).cast(dtypes.float32)

				blocks = (dl * q).reshape(n_blocks, QK_K)
				
			elif self._qtype == GGMLQuantizationType.Q4_0:
				d, qs = hsplit(blocks, [2])

				d = _convert_f16_to_f32(d)

				# qs = qs.reshape(n_blocks, -1, 1, cls.block_size // 2) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
				qs = broadcast_rshift(qs.reshape(n_blocks, -1, 1, self._block_size // 2), [0, 4], 2)
				qs = (qs & 0x0F).reshape(n_blocks, -1).cast(dtypes.int8) - 8

				blocks = (d * qs.cast(dtypes.float32))
			
			elif self._qtype == GGMLQuantizationType.Q4_K:
				d, rest = hsplit(blocks, [2])
				dmin, rest = hsplit(rest, [2])
				scales, qs = hsplit(rest, [12])

				d = _convert_f16_to_f32(d)
				dmin = _convert_f16_to_f32(dmin)

				sc, m = _get_scale_min(scales)

				d = (d * sc.cast(dtypes.float32)).reshape(n_blocks, -1, 1)
				dm = (dmin * m.cast(dtypes.float32)).reshape(n_blocks, -1, 1)

				#qs = qs.reshape(n_blocks, -1, 1, 32) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
				qs = broadcast_rshift(qs.reshape(n_blocks, -1, 1, 32), [0, 4], 2)
				qs = (qs & 0x0F).reshape(n_blocks, -1, 32).cast(dtypes.float32)

				blocks = (d * qs - dm).reshape(n_blocks, QK_K)
				
			elif self._qtype == GGMLQuantizationType.Q5_K:
				d, rest = hsplit(blocks, [2])
				dmin, rest = hsplit(rest, [2])
				scales, rest = hsplit(rest, [12])
				qh, qs = hsplit(rest, [QK_K // 8])

				d = _convert_f16_to_f32(d)
				dmin = _convert_f16_to_f32(dmin)
				
				# come back to this
				sc, m = _get_scale_min(scales)

				d = (d * sc.cast(dtypes.float32)).reshape(n_blocks, -1, 1)
				dm = (dmin * m.cast(dtypes.float32)).reshape(n_blocks, -1, 1)

				ql = broadcast_rshift(qs.reshape(n_blocks, -1, 1, 32), [0, 4], 2)
				qh = broadcast_rshift(qh.reshape(n_blocks, -1, 1, 32), np.arange(8), 2)
				ql = (ql & 0x0F).reshape(n_blocks, -1, 32)
				qh = (qh & 0x01).reshape(n_blocks, -1, 32)
				q = (ql | (qh << 4)).cast(dtypes.float32)

				blocks = (d * q - dm).reshape(n_blocks, QK_K)
			elif self._qtype == GGMLQuantizationType.Q6_K:
				ql, rest = hsplit(blocks, [QK_K // 2])
				qh, rest = hsplit(rest, [QK_K // 4])
				scales, d = hsplit(rest, [QK_K // 16])
				
				scales = scales.bitcast(dtypes.int8).cast(dtypes.float)
				d = _convert_f16_to_f32(d)
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
				q = (ql | (qh << 4)).cast(dtypes.int8) - 32
				q = q.reshape(n_blocks, QK_K // 16, -1).cast(dtypes.float)
				blocks = (d * q).reshape(n_blocks, QK_K)
			
			else:
				raise NotImplementedError(f"{_get_ggml_qtype_name(self._qtype)} dequantization not yet implemented")
			
			# sanity check present in original gguf code
			assert blocks.dtype == dtypes.float
			assert blocks.shape[-1] == self._block_size
			
			# reshape into proper tensor shape
			self._dequantized = blocks.reshape(*quant_shape_from_byte_shape(shape, self._qtype) )
		
		else:
			raise NotImplementedError(f"Dequantization not implemented for {self._qtype}")
		assert not self._dequantized is None
		return self._dequantized
		
	def __getattr__(self, attr):
		"""
		@public
		Provides a wrapper that allows
		```
		self.attribute
		```
		
		to be treated as
		
		```
		self.dequantize().attribute
		```
		"""
		if hasattr(self._tg, attr):
			return getattr(self.dequantize(), attr)
		else:
			raise AttributeError
	

def _get_ggml_qtype_name(qtype):
	for k, v in GGMLQuantizationType.__dict__.items():
		if v == qtype:
			return k
	raise ValueError(f"No GGML type with enum value {qtype}")
	
