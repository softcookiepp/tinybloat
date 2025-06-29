# Some functions in vanilla tinygrad seem to generate problematic kernels
# that sometimes do not compile.
# Therefore, some will be reimplemented here.
import tinygrad
import numpy as np
from typing import Union
from .complex_tensor import ComplexTensor
from .compatibility import convert_fp8

def min(inp, dim = None, axis = None, keepdim = False, always_return_argmin = False):
	if dim is None:
		dim = axis
	if dim is None:
		# tensor needs to be flattened
		inp = inp.reshape(-1)
		dim = 0
	# The strategy will be to split inp into chunks,
	# get the argmax of the given chunk, then
	# do some goodie stuffs that i kinda forget
	
	# So, we need to get the chunk count.
	chunk_count = inp.shape[dim] - 1
	while chunk_count > 0:
		if inp.shape[dim] % chunk_count == 0:
			break
		chunk_count -= 1
	if chunk_count == 0:
		raise ValueError
	chunk_size = inp.shape[dim] // chunk_count
	chunks = inp.chunk(chunk_count, dim = dim)
	min_chunks = []
	for i, chunk in enumerate(chunks):
		min_chunk = chunk.contiguous().min(dim, keepdim = True).realize()
		if len(min_chunk.shape) == 0:
			min_chunk = min_chunk.reshape(1)
		min_chunks.append(min_chunk)
	cat_dim = dim
	catted = tinygrad.Tensor.cat(*min_chunks, dim = cat_dim)
	out = catted.min(axis = dim, keepdim = keepdim)
	if len(out.shape) == 0 and (not always_return_argmin):
		return out
	
	true_out = catted.min(axis = dim, keepdim = True)	
	
	# now we need the argmax...
	mask = np.array(inp.shape) != np.array(true_out.shape)
	a_shape = []
	for item in mask.astype(int) * np.array(inp.shape) + (mask == False).astype(int):
		a_shape.append(int(item))
	
	a = tinygrad.Tensor.arange(inp.shape[dim], device = inp.device).reshape(*a_shape)
	
	# This will not take duplicate values into account. Might want to change this later :c
	arg_min = ((true_out == inp).cast(tinygrad.dtypes.int)*a).sum(dim, keepdim = keepdim)
	return out, arg_min

def max(inp, dim = None, axis = None, keepdim = False, always_return_argmax = False):
	if dim is None:
		dim = axis
	if dim is None:
		# tensor needs to be flattened
		inp = inp.reshape(-1)
		dim = 0
	# The strategy will be to split inp into chunks,
	# get the argmax of the given chunk, then
	# do some goodie stuffs that i kinda forget
	
	# So, we need to get the chunk count.
	chunk_count = inp.shape[dim] - 1
	while chunk_count > 0:
		if inp.shape[dim] % chunk_count == 0:
			break
		chunk_count -= 1
	if chunk_count == 0:
		raise ValueError
	chunk_size = inp.shape[dim] // chunk_count
	chunks = inp.chunk(chunk_count, dim = dim)
	max_chunks = []
	for i, chunk in enumerate(chunks):
		max_chunk = chunk.contiguous().max(dim, keepdim = True).realize()
		if len(max_chunk.shape) == 0:
			max_chunk = max_chunk.reshape(1)
		max_chunks.append(max_chunk)
	cat_dim = dim
	catted = tinygrad.Tensor.cat(*max_chunks, dim = cat_dim)
	out = catted.max(axis = dim, keepdim = keepdim)
	if len(out.shape) == 0 and (not always_return_argmax):
		return out
	
	true_out = catted.max(axis = dim, keepdim = True)	
	
	# now we need the argmax...
	mask = np.array(inp.shape) != np.array(true_out.shape)
	a_shape = []
	for item in mask.astype(int) * np.array(inp.shape) + (mask == False).astype(int):
		a_shape.append(int(item))
	
	a = tinygrad.Tensor.arange(inp.shape[dim], device = inp.device).reshape(*a_shape)
	
	# This will not take duplicate values into account. Might want to change this later :c
	arg_max = ((true_out == inp).cast(tinygrad.dtypes.int)*a).sum(dim, keepdim = keepdim)
	return out, arg_max

def argmax(inp, dim = None, axis = None, keepdim = False):
	_max, _argmax = max(inp, dim, axis, keepdim, True)
	return _argmax

def argmin(inp, dim = None, axis = None, keepdim = False):
	_min, _argmin = min(inp, dim, axis, keepdim, True)
	return _argmin

def cast(t: Union[tinygrad.Tensor, ComplexTensor], dt: tinygrad.dtype.DType):
	"""
	Safely cast tensor types even if the given backend does not support it.
	"""
	# if it isn't supported on the target device, there is no point in continuing
	dev = t.device
	assert tinygrad.device.is_dtype_supported(dt, dev)
	if tinygrad.device.is_dtype_supported(t.dtype, dev):
		return t.cast(dt)
	elif t.dtype in tinygrad.dtypes.fp8s:
		
		return convert_fp8(t, dt)
		
	return t.to("CPU").cast(dt).to(dev)
