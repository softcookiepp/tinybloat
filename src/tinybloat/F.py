"""
Functions either not present in tinygrad, or are present but have extra capability.
"""
import tinygrad
from .common import assert_same_device
import numpy as np
import math

def group_norm(x, num_groups, weight = None, bias = None, eps = 1.0e-5):
	# derived from the tinygrad source code c:
	assert_same_device(x.device, weight, bias)
	x = x.reshape(x.shape[0], num_groups, -1).layernorm(eps=eps).reshape(x.shape)

	if weight is None or bias is None: return x
	# elementwise_affine on channels
	return x * weight.reshape(1, -1, *[1] * (x.ndim-2)) + bias.reshape(1, -1, *[1] * (x.ndim-2))

def scaled_dot_product_attention(query, key, value, attn_mask=None,
		dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
	assert_same_device(query.device, key, value, attn_mask)
	if enable_gqa:
		# not ever sure what this is
		raise NotImplementedError
	if not scale is None:
		# divide the custom scale factor by the internal scale factor
		internal_scale = 1.0 / np.sqrt(query.size(-1))
		scale_factor = scale/internal_scale
		
		# then multiply the query by it
		query = query * scale_factor
	return query.scaled_dot_product_attention(key, value, attn_mask,
		dropout_p, is_causal)

def gelu(x, approximation = None):
	if approximation is None:
		# use actual gelu
		return x*(1 + (x / math.sqrt(2) ).erf() ) / 2
	elif approximation == "tanh":
		# Tinygrad uses the tanh approximation internally
		return x.gelu()
	raise ValueError(f"approximation must be either None or \"tanh\", got {approximation}")

def interpolate(inp,
		size=None,
		scale_factor=None,
		mode='nearest',
		align_corners=None,
		recompute_scale_factor=None,
		antialias=False
		):
	assert isinstance(inp, tinygrad.Tensor)
	if align_corners is None:
		align_corners = False
	if recompute_scale_factor is True:
		# not dealing with this crap for now
		raise NotImplementedError
	if antialias:
		# or this crap either lol
		raise NotImplementedError
	
	size = list(inp.shape)
	len_size = len(size)
	if not scale_factor is None:
		if isinstance(scale_factor, tuple):
			assert len(scale_factor) == len(size) - 2
			for i, sf in enumerate(scale_factor):
				size[i+2] = int(size[i+2]*scale_factor)
		else:
			for i in range(len_size - 2):
				size[i+2] = int(size[i+2] * scale_factor)
		size = tuple(size)
	else:
		assert isinstance(size, tuple)
	return inp.interpolate(size, mode, align_corners)
