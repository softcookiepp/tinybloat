import tinygrad
from .internal import get_tensor_memoryview

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

def probe_tg_dtypes(tg_device: str):
	supported_dtypes = []
	unsupported_dtypes = []
	for dt in iter_tg_dtypes():
		if dt == tinygrad.dtypes.void:
			# torch has no void, and almost no backends support handling it directly
			continue
		
		# this is where we probe
		if is_dtype_supported(dt, tg_device):
			supported_dtypes.append(dt)
		else:
			unsupported_dtypes.append(dt)
	return supported_dtypes, unsupported_dtypes

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
		
		value[exponent == 0] = (mantissa / 8.0) * (2 ** (1 - bias))
		value[(exponent == 0xF and mantissa == 0)] = np.inf
		value[(exponent == 0xF and mantissa != 0)] = np.nan
		value = value*(sign.astype(np.float32)*(-2) + 1)
	elif fp8_tensor.dtype == tinygrad.dtypes.fp8e5m2:
		sign = ((fp8_np >> 7) & 0x1).astype(np.float32)
		exponent = ((fp8_np >> 2) & 0x1F).astype(np.float32)
		mantissa = (fp8_np & 0x3).astype(np.float32)
		bias = 15
		
		# start with the default
		value = ( (1 + mantissa / 4.0) * (2 ** (exponent - bias)) ).astype(np.float32)
		
		value[exponent == 0] = (mantissa / 4.0) * (2 ** (1 - bias))
		value[(exponent == 0x1F and mantissa == 0)] = np.inf
		value[(exponent == 0x1F and mantissa != 0)] = np.nan
		value = value*(sign.astype(np.float32)*(-2) + 1)
	else:
		raise ValueError
	return tinygrad.Tensor(value, device = fp8_tensor.device).cast(dtype).reshape(fp8_tensor.shape)
