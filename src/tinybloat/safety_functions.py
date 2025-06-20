# Some functions in vanilla tinygrad seem to generate problematic kernels
# that sometimes do not compile.
# Therefore, some will be reimplemented here.
import tinygrad
import numpy as np

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
