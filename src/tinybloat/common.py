import tinygrad
import numpy as np

def _slice_to_square(t, offset = 0):
	if len(t.shape) == 1:
		return t
	n = min(t.shape)
	if offset >= 0:
		return t[0:n - offset, offset:]
	else:
		return t[0 - offset:, 0: offset]

def diag(t, *args, **kwargs):
	offset = 0
	if "diagonal" in kwargs.keys():
		offset = kwargs["diagonal"]
	elif len(args) > 0:
		offset = args[0]
	t_original = t
	t = _slice_to_square(t, offset)
	e = tinygrad.Tensor.eye(t.shape[0], dtype = t.dtype, device = t.device)
	if len(t.shape) == 1:
		# make diagonal matrix from 1-D tensor
		out = t.expand( (t.shape[0], t.shape[0]) ) * e
		if offset < 0:
			out = out.pad( (0, abs(offset), abs(offset), 0) )
		elif offset > 0:
			# pad
			out = out.pad( (offset, 0, 0, offset) )
		return out
	elif len(t.shape) == 2:
		# make 1-D array from 2-D tensor
		out = (t*e).sum(0, keepdim = False).squeeze()
		if len(out.shape) == 0:
			out = out.unsqueeze(-1)
		return out
	else:
		raise RuntimeError(f"Expected 2D or 1D tensor, but got {len(t.shape) }D instead.")


def recursive_get_attribute(obj, key):
	"""
	@private
	"""
	key_terms = key.split(".")
	this_key = key_terms[0]
	try:
		kint = int(this_key)
		val = obj[kint]
	except ValueError:
		val = obj.__getattribute__(this_key)
	if len(key_terms[1:]) > 0:
		remaining_keys = ".".join(key_terms[1:])
		return recursive_get_attribute(val, remaining_keys)
	return val

def is_tinygrad_module(obj):
	"""
	Utility method for checking if an object contains tinygrad tensors.
	
	Returns True if yes, False if no.
	"""
	# here we check if the object has any tinygrad tensors
	state_dict = tinygrad.nn.state.get_state_dict(obj)
	return len(state_dict.keys() ) > 0
	
def module_on_device(obj, device: str):
	"""
	Returns True if all tinygrad.Tensor members of obj are on device, False if otherwise
	"""
	# TODO: canonicalize device
	state_dict = tinygrad.nn.state.get_state_dict(obj)
	assert not len(state_dict.keys() ) == 0, "Not a tinygrad module"
	for k, v in state_dict.items():
		if not v.device == device:
			return False
	return True

def move_to_device(obj, device: str):
	"""
	Moves all tinygrad.Tensor instance members in an obj to device
	"""
	for k, v in tinygrad.nn.state.get_state_dict(obj).items():
		v.replace(v.to(device) )
	return obj
	
def cast_to_dtype(obj, dtype: tinygrad.dtype.DType):
	for k, v in tinygrad.nn.state.get_state_dict(obj).items():
		v.replace(v.cast(dtype) )
	return obj

def limit_float_precision(obj, low: Union[tinygrad.dtype.DType, None], high: Union[tinygrad.dtype.DType, None]):
	"""
	Limit the precision of a given module's floating point tensors
	"""
	raise NotImplementedError

def limit_int_precision(obj, low: Union[tinygrad.dtype.DType, None], high: Union[tinygrad.dtype.DType, None]):
	"""
	Limit the precision of a given module's integer tensors.
	"""
	raise NotImplementedError
	
def nonzero(inp, as_tuple = False):
	# It is going to be very difficult to write this function
	# in a manner that is JIT-compatible :c
	# since the shapes can be variable-length :c
	out = np.nonzero(inp.numpy() )
	if as_tuple:
		raise NotImplementedError
	out_to_cat = []
	for idx_a in out:
		out_to_cat.append(idx_a.reshape(-1, 1) )
	return tinygrad.Tensor(np.concatenate(out_to_cat, axis = 1), device = inp.device)
	
	
def cat(tensors, dim = 0):
	# is this even necessary??
	tbase = tensors[0]
	trest = tuple(tensors[1:])
	assert_same_device(tbase.device, trest)
	return tbase.cat(*trest, dim = dim)

def cumprod(inp, dim, dtype=None, out=None):
	# first, get the slices used in the __getitem__ call for each element
	slices = []
	for i in range(len(inp.shape)):
		slices.append(slice(None, None, None) )
	
	outputs = []
	for i in range(inp.shape[dim] ):
		slices[dim] = slice(0, i + 1, None)
		new_shape = list(inp.shape)
		new_shape[dim] = -1
		new_shape = tuple(new_shape)
		outputs.append(inp[slices].prod(dim).reshape(new_shape) )
	return cat(outputs, dim)

def assert_same_device(dev, *inp):
	dev = tinygrad.Device.canonicalize(dev)
	if len(inp) == 1:
		inp = inp[0]
	if hasattr(inp, "tg"):
		assert dev == inp.tg.device
	if isinstance(inp, tinygrad.Tensor):
		assert dev == inp.device
	elif isinstance(inp, list) or isinstance(inp, tuple):
		for item in inp:
			assert_same_device(dev, item)
	elif isinstance(inp, dict):
		for k, v in inp.items():
			assert_same_device(dev, v)
		return inp
	else:
		if hasattr(inp, "__dict__"):
			# treat as dictionary hehe
			assert_same_device(dev, inp.__dict__)
			
def chunk(inp, chunks: int, dim: int = 0):
	return inp.chunk(chunks, dim)
	
def clamp(inp, min = None, max = None):
	return inp.clamp(min, max)
	
def stack(tensors, dim = 0, out = None):
	assert out is None
	tbase = tensors[0]
	trest = tuple(tensors[1:])
	assert_same_device(tbase.device, trest)
	return tbase.stack(*trest, dim = dim)

def outer(u, v):
	assert len(u.shape) == len(v.shape) == 1, "Both supplied tensors must be 1D"
	u_expanded = u.reshape(-1, 1).expand(-1, v.shape[0])
	v_expanded = v.reshape(1, -1).expand(u.shape[0], -1)
	return u_expanded * v_expanded
