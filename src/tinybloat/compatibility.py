import tinygrad

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
