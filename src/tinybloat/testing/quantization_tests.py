#import gadget
#from gadget import ggml, GgmlModel
from huggingface_hub import hf_hub_download

import tinybloat
import tinygrad
import torch

from .testing_utils import *
from .testing_utils import _test_function, _test_hf_reimplementation
import numpy as np

# hopefully this should work...
import diffusers
import gguf

def _test_matmul():
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


def test_dequantize():
	for method in ["Q2_K", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q4_0", "Q6_K"]:
		path = hf_hub_download(repo_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", filename = f"tinyllama-1.1b-chat-v1.0.{method}.gguf")
		reader = gguf.GGUFReader(path)
		for tensor in reader.tensors:
			out = gguf.dequantize(tensor.data, tensor.tensor_type)
			
			quantized = tinygrad.Tensor(np.array(tensor.data) )
			
			# just make sure it actually loads the tensor correctly
			assert mse(quantized.numpy(), tensor.data) < 1.0e-4
			assert mse(np.array(quantized.shape), np.array(tensor.data.shape) ) < 1.0e-4
			
			qt = tinybloat.quantization.QTensor(quantized, tensor.tensor_type)
			assert mse(qt.dequantize().numpy(), out) < 1.0e-4
