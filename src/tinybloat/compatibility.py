import tinygrad
from .internal import get_tensor_memoryview
import numpy as np
from typing import Union, Optional, Tuple

_longlong_support_status = {}

def _device_supports_longlong(dev: str) -> bool:
	# The test will make a big enough tensor that requires long long to index
	# If a compile error is thrown, then we return false.
	# If it succeeds, we return true
	big_shp = (1, 77, 49408, 768)
	a = tinygrad.Tensor.randn(big_shp)
	try:
		a.realize()
	except tinygrad.device.CompileError:
		del a
		return False
	del a
	return True

def device_supports_longlong(dev: str) -> bool:
	if dev in _longlong_support_status.keys():
		return _longlong_support_status[dev]
	return _device_supports_longlong(dev)

_dtype_table = {
	# device, supported, uns
}

def _test_dtype(dtype, device):
	a = tinygrad.Tensor.randn(4, dtype = dtype, device = device).realize()
	return True

def _probe_tg_dtypes(tg_device: str):
	"""
	Doing this because I straight-up don't trust tinygrad to be accurate about this
	"""
	supported_dtypes = []
	unsupported_dtypes = []
	for dt in iter_tg_dtypes():
		if dt == tinygrad.dtypes.void:
			# does this even matter?
			continue
		
		# this is where we probe
		if _test_dtype(dt, tg_device):
			supported_dtypes.append(dt)
		else:
			unsupported_dtypes.append(dt)
	return supported_dtypes, unsupported_dtypes

def device_supports_dtype(device: str, dtype):
	if device in _dtype_table.keys():
		return dtype in _dtype_table[device]
	_dtype_table[device] = _probe_tg_types(tg_device)[0]
	return dtype in _dtype_table[device]
	
def get_device_float_bounds(device: str):
	raise NotImplementedError
	
def get_device_sint_bounds(device: str):
	raise NotImplementedError

def get_device_uint_bounds(device: str):
	raise NotImplementedError
	
def assert_dtype_supported(device: str, dtype: tinygrad.dtype.DType):
	raise NotImplementedError
	
def convert_fp8(fp8_tensor, dtype):
	"""
	Convert a fp8 tensor to another dtype even if no backends on system support it
	"""
	mem = get_tensor_memoryview(fp8_tensor)
	fp8_np = np.array(mem)
	value = None
	if fp8_tensor.dtype == tinygrad.dtypes.fp8e4m3:
		sign = ((fp8_np >> 7) & 0x1).astype(np.float32)
		exponent = ((fp8_np >> 3) & 0xF).astype(np.float32)
		mantissa = (fp8_np & 0x7).astype(np.float32)
		bias = 7
		
		# start with the default
		value = ( (1 + mantissa / 8.0) * (2 ** (exponent - bias)) ).astype(np.float32)
		exponent_nonzero_idx = np.nonzero((exponent == 0)*np.arange(len(value) ) )[0].astype(int)
		if exponent_nonzero_idx.size > 0:
			value[exponent_nonzero_idx] = (mantissa / 8.0) * (2 ** (1 - bias))
		inf_idx = np.nonzero( ( (exponent == 0xF) * (mantissa == 0) )*np.arange(len(value) ) )[0].astype(int)
		if inf_idx.size > 0:
			value[ind_idx] = np.inf
		nan_idx = np.nonzero( ( (exponent == 0xF) * (mantissa != 0) )*np.arange(len(value) ) )[0].astype(int)
		if nan_idx.size > 0:
			value[nan_idx] = np.nan
		value = value*(sign.astype(np.float32)*(-2) + 1)
	elif fp8_tensor.dtype == tinygrad.dtypes.fp8e5m2:
		sign = ((fp8_np >> 7) & 0x1).astype(np.float32)
		exponent = ((fp8_np >> 2) & 0x1F).astype(np.float32)
		mantissa = (fp8_np & 0x3).astype(np.float32)
		bias = 15
		
		# start with the default
		value = ( (1 + mantissa / 4.0) * (2 ** (exponent - bias)) ).astype(np.float32)
		
		exponent_nonzero_idx = np.nonzero((exponent == 0)*np.arange(len(value) ) )[0].astype(int)
		if exponent_nonzero_idx.size > 0:
			value[exponent_nonzero_idx] = (mantissa / 4.0) * (2 ** (1 - bias))
		inf_idx = np.nonzero( (exponent == 0x1F)*(mantissa == 0)*np.arange(len(value) ) )[0].astype(int)
		if inf_idx.size > 0:
			value[ind_idx] = np.inf
		nan_idx = np.nonzero( (exponent == 0x1F)*(mantissa != 0)*np.arange(len(value) ) )[0].astype(int)
		if nan_idx.size > 0:
			value[nan_idx] = np.nan
		value = value*(sign.astype(np.float32)*(-2) + 1)
	else:
		raise ValueError
	return tinygrad.Tensor(value, device = fp8_tensor.device).cast(dtype).reshape(fp8_tensor.shape)
