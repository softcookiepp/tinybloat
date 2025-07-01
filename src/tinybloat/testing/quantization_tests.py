import gadget
from gadget import ggml, GgmlModel

import tinybloat
import tinygrad
import torch

from .testing_utils import *
from .testing_utils import _test_function, _test_hf_reimplementation
import numpy as np

# hopefully this should work...
import diffusers

def test_matmul():
	"""
	Just a basic test to make sure we are doing this stuff correctly
	"""
	class MatmulTest(GgmlModel):
		A: gadget.Tensor("F32", (4, 4) )
		B: gadget.Tensor("F32", (4, 4) )
		
		def forward(self):
			ctx = self.ctx_graph
			A, B = self.tensors["A", "B"]
			return ggml.ggml_mul_mat(ctx, A, B)
	
	mm = MatmulTest.from_values({})
	a = np.arange(16).reshape(4, 4).astype(np.float32)
	b = np.arange(16).reshape(4, 4).astype(np.float32)
	g_output = mm(A = a, B = np.ascontiguousarray(b.T) ).T
	error = mse(g_output, a @ b)
	print(a)
	print(a @ b)
	print(g_output)
	assert error < 1.0e-4, f"error too high: {error}"

def test_bitcast():
	
	def do_bitcast(a):
		if isinstance(a, torch.Tensor):
			return a.view(torch.int8) << 4
		return (a.bitcast(tinygrad.dtypes.uint8) << 4).bitcast(tinygrad.dtypes.int8)
	
	d = make_test_data(40)
	_test_function([d], {}, do_bitcast, do_bitcast)


def test_quantize_4_0():
	raise NotImplementedError
