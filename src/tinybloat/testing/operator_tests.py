from .testing_utils import *
from .testing_utils import _test_function

import torch
import tinybloat

def test_cumprod():
	inp = np.arange(5*4).reshape(5, 4).astype(np.float32)
	_test_function( (inp, 0), {}, torch.cumprod, tinybloat.cumprod)


def test_cat():
	a = make_test_data(40, 2, 5)
	b = make_test_data(40, 2, 5)
	for i in range(3):
		_test_function( ([a, b], i), {}, torch.cat, tinybloat.cat)
	_test_function( ([a, b], -1), {}, torch.cat, tinybloat.cat)

def test_interpolate():
	shape = (2, 3, 6, 6)
	a = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
	args = (a, None, 2.0)
	_test_function(args, {}, torch_function = torch.nn.functional.interpolate, tinygrad_function = tinybloat.F.interpolate)

def _test_unary(torch_function, tinygrad_function, data = None):
	if data is None:
		shape = (4, 2, 6, 8)
		data = make_test_data(*shape)
	_test_function( (data), {}, torch_function, tinygrad_function)
	

def test_scaled_dot_product_attention():
	q = make_test_data(1, 8, 4096, 40)
	k = make_test_data(1, 8, 4096, 40)
	v = make_test_data(1, 8, 4096, 40)
	_test_function( [q, k, v], {}, torch.nn.functional.scaled_dot_product_attention, tinybloat.F.scaled_dot_product_attention)

def test_gelu():
	_test_unary(torch.nn.functional.gelu, tinybloat.F.gelu, np.arange(1000).astype(np.float32) - 500.0 )

def _test_chunk(dim):	
	data = make_test_data(16, 8, 4, 8)
	_test_function([data, 2, dim], {}, torch.chunk, tinybloat.chunk)

def test_chunk():
	for i in range(4):
		_test_chunk(i)
	_test_chunk(-1)

def test_clamp():
	data = make_test_data(3, 5, 8)
	_test_function([data, 0.0, 0.5], {}, torch.clamp, tinybloat.clamp)

def test_stack():
	tensors = []
	for i in range(3):
		tensors.append(make_test_data(4, 4, 4) )
	
	for i in range(3):
		_test_function( [tensors, i], {}, torch.stack, tinybloat.stack )

def test_max():
	a = make_test_data(3, 2, 5, 8)
	_test_function([a], {}, torch.max, tinybloat.safety_functions.max)

def test_normal_():
	a = np.arange(16384).astype(np.float32)
	def test_normal__impl(t):
		old_t = t
		if isinstance(t, torch.Tensor):
			# torch
			t = torch.nn.init.normal_(t)
			return torch.mean(old_t), torch.var(old_t), torch.mean(t), torch.var(t)
		else:
			# adapter tensor
			t = tinybloat.nn.init.normal_(t)
			return old_t.mean(), old_t.var(), t.mean(), t.var()
	_test_function([a], {}, test_normal__impl, test_normal__impl, error_threshold = 1.0e-2)
		
def test_diag():
	for i in range(2):
		shape = []
		for i2 in range(i + 1):
			shape.append(int(4) )
		shape = tuple(shape)
		a = np.arange(np.prod(shape) ).reshape(shape).astype(np.float32)
		for diagonal in range(3*2):
			diagonal = diagonal - 3
			_test_function([a, diagonal], {}, torch.diag, tinybloat.diag)
			
def test_norm():
	for dim in range(2):
		shape = []
		for i2 in range(dim + 1):
			shape.append(int(4) )
		shape = tuple(shape)
		a = np.arange(np.prod(shape) ).reshape(shape).astype(np.float32)
		for keepdim in [True, False]:
			_test_function([a], {"dim": dim, "keepdim": keepdim}, torch.linalg.norm, tinybloat.linalg.norm)

def test_outer():
	u = make_test_data(5)
	v = make_test_data(8)
	_test_function([u, v], {}, torch.outer, tinybloat.outer)
			
def test_qr():
	A = np.random.randn(4*4).reshape(4, 4).astype(np.float32)
	
	# actual underlying q value seems to differ, but does that matter?
	def _qr_test(a):
		if isinstance(a, torch.Tensor):
			q, r = torch.linalg.qr(a)
		else:
			q, r = tinybloat.linalg.qr(a)
		return q@r
	_test_function([A], {}, _qr_test, _qr_test)
	
def test_complex_add():
	a = make_test_data(16) + 1.0j*make_test_data(16)
	b = make_test_data(16) + 1.0j*make_test_data(16)
	add_test = lambda x, y: x + y
	_test_function([a, b], {}, add_test, add_test)
	
def test_complex_sub():
	a = make_test_data(16) + 1.0j*make_test_data(16)
	b = make_test_data(16) + 1.0j*make_test_data(16)
	sub_test = lambda x, y: x - y
	_test_function([a, b], {}, sub_test, sub_test)
	
def test_complex_mul():
	a = make_test_data(16) + 1.0j*make_test_data(16)
	b = make_test_data(16) + 1.0j*make_test_data(16)
	mul_test = lambda x, y: x * y
	_test_function([a, b], {}, mul_test, mul_test)
	
def test_complex_div():
	a = make_test_data(16) + 1.0j*make_test_data(16)
	b = make_test_data(16) + 1.0j*make_test_data(16)
	div_test = lambda x, y: x / y
	_test_function([a, b], {}, div_test, div_test)
	
def test_complex_rsub():
	a = make_test_data(16) + 1.0j*make_test_data(16)
	b = make_test_data(16) + 1.0j*make_test_data(16)
	rsub_test = lambda x, y: 10 - y
	_test_function([a, b], {}, rsub_test, rsub_test)
	
def test_complex_rdiv():
	a = make_test_data(16) + 1.0j*make_test_data(16)
	b = make_test_data(16) + 1.0j*make_test_data(16)
	rdiv_test = lambda x, y: 10 / y
	_test_function([a, b], {}, rdiv_test, rdiv_test)

def test_complex_matmul():
	a = make_test_data(16).reshape(4, 4) + 1.0j*make_test_data(16).reshape(4, 4)
	b = make_test_data(16).reshape(4, 4) + 1.0j*make_test_data(16).reshape(4, 4)
	matmul_test = lambda x, y: x @ y
	_test_function([a, b], {}, matmul_test, matmul_test)
	
def _zeros_like(inp):
	if isinstance(inp, torch.Tensor):
		return torch.zeros_like(inp)
	else:
		return tinybloat.zeros_like(inp)

def test_eig():
	A = np.array([ [1, 1], [0, 2] ]).reshape(2, 2).astype(np.float32)
	def _eig_test(a):
		if isinstance(a, torch.Tensor):
			result = torch.linalg.eig(a)
			vals, vecs = result
			vals = vals.real
			vecs = vecs.real
		else:
			vals, vecs = tinybloat.linalg.eig(a)
		return vals, vecs
		n = vecs.shape[0]
		comparisons = []
		for i in range(n):
			val = vals[i]
			vec = vecs[:, i]
			q = a @ vec.reshape(-1, 1)
			b = val * vec
			
			# MSE doesn't handle complex numbers well :c
			try:
				q_imag = q.imag
			except RuntimeError:
				# just do 0
				q_imag = _zeros_like(q.real)
				
			try:
				b_imag = b.imag
			except RuntimeError:
				# just do 0
				b_imag = _zeros_like(b.real)
			comparisons.append((q.real, q_imag, b.real, b_imag))
		return comparisons
			
	#_test_function([A], {}, torch.linalg.eig, tinybloat.linalg.eig)
	_test_function([A], {}, _eig_test, _eig_test)
	

def test_max():
	a = make_test_data(4, 8)
	def _test_max(t, *args, **kwargs):
		if isinstance(t, torch.Tensor):
			out = torch.max(t, *args, **kwargs)
		else:
			out = tinybloat.max(t, *args, **kwargs)
		return out
		
	_test_function([a], {}, _test_max, _test_max)
	for dim in [0, 1]:
		_test_function([a], {"axis": dim}, _test_max, _test_max)
		
def test_argmax():
	a = make_test_data(4, 8)
	def _test_argmax(t, *args, **kwargs):
		if isinstance(t, torch.Tensor):
			out = torch.argmax(t, *args, **kwargs)
		else:
			out = tinybloat.argmax(t, *args, **kwargs)
		return out
		
	_test_function([a], {}, _test_argmax, _test_argmax)
	for dim in [0, 1]:
		_test_function([a], {"axis": dim}, _test_argmax, _test_argmax)
		
def test_nonzero():
	a = np.arange(16).reshape(4, 4).astype(np.float32) - 4
	f = lambda x: x.nonzero()
	def f(t, *args, **kwargs):
		if isinstance(t, torch.Tensor):
			return t.nonzero()
		return tinybloat.nonzero(t)
		
	_test_function([a], {}, f, f)

def test_limit_float_precision():
	class DummyModule:
		def __init__(self):
			self.too_high = tinygrad.Tensor.arange(4, device = "CPU").cast(tinygrad.dtypes.double)
			self.too_low = tinygrad.Tensor.arange(4, device = "CPU").cast(tinygrad.dtypes.half)
			
	dummy = DummyModule()
	tinybloat.limit_float_precision(dummy, tinygrad.dtypes.float, tinygrad.dtypes.float)
	for k, v in tinygrad.nn.state.get_state_dict(dummy).items():
		assert v.dtype == tinygrad.dtypes.float, f"here is the dtype: {v.dtype}"

def test_limit_sint_precision():
	class DummyModule:
		def __init__(self):
			self.too_high = tinygrad.Tensor.arange(4, device = "CPU").cast(tinygrad.dtypes.int64)
			self.too_low = tinygrad.Tensor.arange(4, device = "CPU").cast(tinygrad.dtypes.int8)
			
	dummy = DummyModule()
	tinybloat.limit_sint_precision(dummy, tinygrad.dtypes.int16, tinygrad.dtypes.int32)
	assert dummy.too_high.dtype == tinygrad.dtypes.int32
	assert dummy.too_low.dtype == tinygrad.dtypes.int16

def test_limit_uint_precision():
	class DummyModule:
		def __init__(self):
			self.too_high = tinygrad.Tensor.arange(4, device = "CPU").cast(tinygrad.dtypes.uint64)
			self.too_low = tinygrad.Tensor.arange(4, device = "CPU").cast(tinygrad.dtypes.uint8)
			
	dummy = DummyModule()
	tinybloat.limit_uint_precision(dummy, tinygrad.dtypes.uint16, tinygrad.dtypes.uint32)
	assert dummy.too_high.dtype == tinygrad.dtypes.uint32
	assert dummy.too_low.dtype == tinygrad.dtypes.uint16
	
def test_convert_fp8_safely():
	test_b_array = bytearray("123145356234", "utf-8")
	test_b = b"123145356234"
	
	# first do it with e4m3
	ttorch = torch.frombuffer(test_b_array, dtype = torch.float8_e4m3fn)
	ttiny = tinygrad.Tensor(test_b, dtype = tinygrad.dtypes.fp8e4m3)
	ttorch = ttorch.to(torch.float32)
	ttiny = tinybloat.safety_functions.cast(ttiny, tinygrad.dtypes.float32)
	assert mse(ttorch.numpy(), ttiny.numpy() ) == 0.0
	
	# then with e5m2
	ttorch = torch.frombuffer(test_b_array, dtype = torch.float8_e5m2)
	ttiny = tinygrad.Tensor(test_b, dtype = tinygrad.dtypes.fp8e5m2)
	ttorch = ttorch.to(torch.float32)
	ttiny = tinybloat.safety_functions.cast(ttiny, tinygrad.dtypes.float32)
	assert mse(ttorch.numpy(), ttiny.numpy() ) == 0.0
	
def test_device_supports_dtype():
	for dt in tinygrad.dtypes.all:
		tinybloat.compatibility.device_supports_dtype(tinygrad.Device.DEFAULT, dt)
