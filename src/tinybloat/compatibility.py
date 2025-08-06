import tinygrad
from tinygrad import dtypes
from .internal import get_tensor_memoryview
import numpy as np
from typing import Union, Optional, Tuple
import subprocess
import itertools

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

def _recursive_get_uop_list(uop):
	uops = []
	uops.append(uop)
	for child in uop.src:
		for new_uop in _recursive_get_uop_list(child):
			uops.append(new_uop)
	return uops
	
def _recursive_get_items_of_type(obj, python_type):
	if isinstance(obj, python_type):
		yield obj
	elif isinstance(obj, tuple) or isinstance(obj, list):
		for item in obj:
			if isinstance(item, python_type):
				yield item
	elif isinstance(obj, dict):
		for item in _recursive_get_items_of_type(list(obj.values() ), python_type):
			yield item
	elif hasattr(obj, "__dict__"):
		for item in _recursive_get_items_of_type(obj.__dict__, python_type):
			yield item

def tensor_requires_longlong(t: tinygrad.Tensor):
	# anything larger than this requires a longlong
	int_max = np.iinfo(np.dtype("int64") ).max
	if np.prod(t.shape) > int_max:
		return True
	
	# we need a uop list in order to parse the arguments to find an int
	uop_list = _recursive_get_uop_list(t.uop)
	args = []
	for uop in uop_list:
		args.append(uop.arg)
	for target in _recursive_get_items_of_type(args, int):
		if target > int_max:
			return True
	return False
	

_dtype_table = {
	# device: [supported]
}

def _test_dtype(dtype, device):
	if tinygrad.device.is_dtype_supported(dtype, device):
		try:
			input(f"testing dtype for {device}: {dtype}")
			a = (tinygrad.Tensor.randn(4, dtype = dtype, device = device).bitcast(dtype).sin() ).realize().numpy()
			print("success!")
			return True
		except (tinygrad.device.CompileError, subprocess.CalledProcessError, KeyError, RuntimeError) as e:
			return False
	return False

def _probe_tg_dtypes(tg_device: str):
	"""
	Doing this because I straight-up don't trust tinygrad to be accurate about this
	"""
	supported_dtypes = []
	unsupported_dtypes = []
	for dt in tinygrad.dtypes.all:
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
	_dtype_table[device] = _probe_tg_dtypes(device)[0]
	return dtype in _dtype_table[device]
	
def _get_device_type_group_bounds(device: str, dt_list):
	low = None
	assert len(dt_list) > 0
	for dt in sorted(dt_list, key = lambda x: x.itemsize):
		if device_supports_dtype(device, dt):
			low = dt
			break
	assert not low is None, f"low not found in {dt_list} for {device}"
	high = None
	for dt in sorted(dt_list, key = lambda x: -1*x.itemsize):
		if device_supports_dtype(device, dt):
			high = dt
			break
	assert not high is None
	return low, high
	
def get_device_float_bounds(device: str):
	return _get_device_type_group_bounds(device, tinygrad.dtypes.floats)
	
def get_device_sint_bounds(device: str):
	return _get_device_type_group_bounds(device, tinygrad.dtypes.sints)

def get_device_uint_bounds(device: str):
	return _get_device_type_group_bounds(device, tinygrad.dtypes.sints)
	
def convert_fp16(fp16_tensor, dtype):
	bias = 15
	val = fp16_tensor.bitcast(dtypes.uint16)
	sign = ((val >> 15) & 0b0000000000000001).cast(dtypes.float)
	exponent = ((val >> 10) & 0b0000000000011111).cast(dtypes.float)
	mantissa = (val & 0b0000001111111111).cast(dtypes.float)
	
	# start with the default
	value = ( (1 + (mantissa/1024) ) * (2 ** (exponent - bias)) ).cast(dtypes.float)
	
	value = (exponent == 0).where(
		(mantissa / 1024) * (2 ** (-14)),
		value
	)
	
	
	value = ( (exponent == 0b11111) * (mantissa != 0) ).where(np.nan, value)
	value = ( (exponent == 0b11111) * (mantissa == 0) ).where(np.inf, value)
	value = value*( (-1)**sign.cast(dtypes.float32))
	return value
	
def convert_fp8e4m3(fp8_tensor, dtype):
	fp8_np = fp8_tensor.bitcast(dtypes.uint8)
	sign = ((fp8_np >> 7) & 0x1).cast(dtypes.float32)
	exponent = ((fp8_np >> 3) & 0xF).cast(dtypes.float32)
	mantissa = (fp8_np & 0x7).cast(dtypes.float32)
	bias = 7
	
	# start with the default
	value = ( (1 + mantissa / 8.0) * (2 ** (exponent - bias)) ).cast(dtypes.float32)
	
	value = (exponent == 0).where(
		(mantissa / 8) * (2 ** (1 - bias)),
		value
	)
	
	value = ( (exponent == 0xF) * (mantissa != 0) ).where(np.nan, value)
	value = ( (exponent == 0xF) * (mantissa == 0) ).where(np.inf, value)
	return ( value*( (-1)**sign.cast(dtypes.float32)) ).cast(dtype)
	
def convert_fp8e5m2(fp8_tensor, dtype):
	fp8_np = fp8_tensor.bitcast(dtypes.uint8)
	sign = ((fp8_np >> 7) & 0x1).cast(dtypes.float32)
	exponent = ((fp8_np >> 2) & 0x1F).cast(dtypes.float32)
	mantissa = (fp8_np & 0x3).cast(dtypes.float32)
	bias = 15
	
	# start with the default
	value = ( (1 + mantissa / 4.0) * (2 ** (exponent - bias)) ).cast(dtypes.float32)
	
	# subnormals
	value = (exponent == 0).where(
		(mantissa / 4) * (2 ** (1 - bias)),
		value
	)
	
	value = ( (exponent == 0x1F) * (mantissa != 0) ).where(np.nan, value)
	value = ( (exponent == 0x1F) * (mantissa == 0) ).where(np.inf, value)
	return ( value*( (-1)**sign.cast(dtypes.float32)) ).cast(dtype)

def convert_fp8(fp8_tensor, dtype):
	"""
	Convert a fp8 tensor to another dtype even if no backends on system support it
	"""
	if fp8_tensor.dtype == tinygrad.dtypes.fp8e4m3:
		return convert_fp8e4m3(fp8_tensor, dtype)
		
	elif fp8_tensor.dtype == tinygrad.dtypes.fp8e5m2:
		return convert_fp8e5m2(fp8_tensor, dtype)
	
def convert_bfloat16(bf16_tensor, dtype):
	bias = 127
	val = bf16_tensor.bitcast(dtypes.uint16)
	sign = ((val >> 15) & 0b0000000000000001).cast(dtypes.float)
	exponent = ((val >> 7) & 0b0000000011111111).cast(dtypes.float)
	mantissa = (val & 0b0000000001111111).cast(dtypes.float)
	
	# start with the default
	value = ( (1 + (mantissa/128) ) * (2 ** (exponent - bias)) ).cast(dtypes.float)
	
	value = (exponent == 0).where(
		(mantissa / 128) * (2 ** (-126)),
		value
	)
	
	
	value = ( (exponent == 0xFF) * (mantissa != 0) ).where(np.nan, value)
	value = ( (exponent == 0xFF) * (mantissa == 0) ).where(np.inf, value)
	return ( value*( (-1)**sign.cast(dtypes.float32)) ).cast(dtype)
