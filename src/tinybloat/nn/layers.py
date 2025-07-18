import tinygrad
from ..common import move_to_device, cast_to_dtype
from .init import xavier_uniform_
from ..compatibility import device_supports_longlong

class MultiheadAttention:
	"""
	Near-exact reimplementation of [torch.nn.MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
	"""
	def __init__(self, embed_dim,
				num_heads,
				dropout=0.0,
				bias=True,
				add_bias_kv=False,
				add_zero_attn=False,
				kdim=None, vdim=None,
				batch_first=False,
				device=None,
				dtype=None
			):
		"""
		:param embed_dim: Total dimension of the model.
		:param num_heads:
		:param dropout:
		:param bias:
		:param add_bias_kv:
		:param add_zero_attn:
		:param kdim:
		:param vdim:
		:param batch_first:
		:param device:
		:param dtype:
		"""
		if embed_dim <= 0 or num_heads <= 0:
			raise ValueError(
				f"embed_dim and num_heads must be greater than 0,"
				f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
			)
		factory_kwargs = {"device": device, "dtype": dtype}
		self.embed_dim = embed_dim
		self.kdim = kdim if kdim is not None else embed_dim
		self.vdim = vdim if vdim is not None else embed_dim
		self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

		self.num_heads = num_heads
		self.dropout = dropout
		self.batch_first = batch_first
		self.head_dim = embed_dim // num_heads
		assert (
			self.head_dim * num_heads == self.embed_dim
		), "embed_dim must be divisible by num_heads"

		if not self._qkv_same_embed_dim:
			self.q_proj_weight = tinygrad.Tensor.zeros((embed_dim, embed_dim), **factory_kwargs)
			self.k_proj_weight = tinygrad.Tensor.zeros((embed_dim, self.kdim), **factory_kwargs)
			self.v_proj_weight = tinygrad.Tensor.zeros((embed_dim, self.vdim), **factory_kwargs)
		else:
			self.in_proj_weight = tinygrad.Tensor.zeros((3 * embed_dim, embed_dim), **factory_kwargs)

		if bias:
			self.in_proj_bias = tinygrad.Tensor.zeros(3 * embed_dim, **factory_kwargs)
		# this may or may not work...
		self.out_proj = tinygrad.nn.Linear(embed_dim, embed_dim, bias=bias)
		if not device is None:
			move_to_device(self.out_proj, device)
		if not dtype is None:
			cast_to_dtype(self.out_proj, dtype)

		if add_bias_kv:
			self.bias_k = tinygrad.Tensor.zeros((1, 1, embed_dim), **factory_kwargs)
			self.bias_v = tinygrad.Tensor.zeros((1, 1, embed_dim), **factory_kwargs)
		else:
			self.bias_k = self.bias_v = None

		self.add_zero_attn = add_zero_attn
		self.max_self_attn_cache_len = 512 # lets just try this lol

	def __call__(self, q, k, v, key_padding_mask = None,
			need_weights = False, attn_mask = None,
			average_attn_weights = False, is_causal = False):
		"""
		`need_weights` is `False` by default, since `tinygrad.Tensor.scaled_dot_product_attention` has no such option.
		As such it is not yet implemented :c
		"""
		if True in (not key_padding_mask is None, average_attn_weights, is_causal):
			# not going to bother with these for now
			raise NotImplementedError
		if hasattr(self, "in_proj_weight"):
			wq, wk, wv = self.in_proj_weight.chunk(3, dim = 0)
		else:
			wq, wk, wv = self.q_proj_weight, self.k_proj_weight, self.v_proj_weight
			
		# YAY!!!!!!
		q = q @ wq.T
		k = k @ wk.T
		v = v @ wv.T
		
		if hasattr(self, "in_proj_bias"):
			q = q + self.in_proj_bias[0:self.embed_dim]
			k = k + self.in_proj_bias[self.embed_dim:self.embed_dim*2]
			v = v + self.in_proj_bias[self.embed_dim*2:self.embed_dim*3]
		
		if not self.bias_k is None:
			b += self.bias_k
			v += self.bias_v
		qc = q.chunk(self.num_heads, dim = 1)
		kc = k.chunk(self.num_heads, dim = 1)
		vc = v.chunk(self.num_heads, dim = 1)
		
		assert qc[0].shape[1] == vc[0].shape[1] == kc[0].shape[1] == self.head_dim
		
		att = []
		weights = []
		for head in range(self.num_heads):
			qi, ki, vi = qc[head], kc[head], vc[head]
			hi = tinygrad.Tensor.scaled_dot_product_attention(qi, ki, vi, attn_mask = attn_mask)
			if need_weights:
				raise NotImplementedError
			att.append(hi)
		weight = tinygrad.Tensor.cat(*att, dim = 1)
		out = self.out_proj(weight)
		if need_weights:
			# For now, weight just miight be inaccurate :c
			return out, weight[0:out.shape[0], 0:out.shape[0]]
		return (out,)


class Embedding:
	"""
	Functions identically to tinygrad.nn.Embedding, but some calculation is offloaded to the CPU if
	the given device does not support tensors of very large size.
	"""
	def __init__(self,
			vocab_size:int,
			embed_size:int
			):
		self.vocab_sz, self.embed_sz = vocab_size, embed_size
		self.weight = tinygrad.Tensor.zeros(vocab_size, embed_size)
		xavier_uniform_(self.weight )
	
	def __call__(self, idx):
		vocab_sz, embed_sz, weight = self.vocab_sz, self.embed_sz, self.weight
		
		original_device = idx.device
		working_device = idx.device
		
		if not device_supports_longlong(weight.device):
			# perform embedding on the CPU as a fallback
			working_device = "CPU"
		
		if not hasattr(self, 'arange'): self.arange = tinygrad.Tensor.arange(vocab_sz,
			requires_grad=False, device=working_device, dtype = highest_precision_int(working_device) ).unsqueeze(-1)
		big_shp = idx.shape+(vocab_sz, embed_sz)
		
		
		idx = idx.to(working_device)
		weight = weight.to(working_device)
		
		arange, idx, vals = parent.arange.expand(big_shp), idx.reshape(idx.shape+(1, 1)).expand(big_shp), weight.expand(big_shp)
		
		# (-1, 77, 49408, -1)
		inter = (arange == idx)
		
		# (-1, 77, 49408, -1)
		inter2 = inter.mul(vals)
		out = inter2.sum(-2)
		
		return out.to(original_device)

