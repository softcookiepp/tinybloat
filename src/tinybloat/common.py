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
	# here we check if the object has any tinygrad tensors
	state_dict = tinygrad.nn.state.get_state_dict(obj)
	return len(state_dict.keys() ) > 0
	
def module_on_device(obj, device: str):
	# TODO: canonicalize device
	state_dict = tinygrad.nn.state.get_state_dict(obj)
	assert not len(state_dict.keys() ) == 0, "Not a tinygrad module"
	for k, v in state_dict.items():
		if not v.device == device:
			return False
	return True

def move_to_device(obj, device: str):
	for k, v in tinygrad.nn.state.get_state_dict(obj).items():
		v.replace(v.to(device) )
	return obj
	
def cast_to_dtype(obj, dtype: tinygrad.dtype.DType):
	for k, v in tinygrad.nn.state.get_state_dict(obj).items():
		v.replace(v.cast(dtype) )
	return obj
	
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
	
	
