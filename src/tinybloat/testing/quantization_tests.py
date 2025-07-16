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
	# the error can be high due to weird float crap sometimes
	assert error < 1.0e-3, f"error too high: {error}"

def test_bitcast():
	
	def do_bitcast(a):
		if isinstance(a, torch.Tensor):
			return a.view(torch.int8) << 4
		return (a.bitcast(tinygrad.dtypes.uint8) << 4).bitcast(tinygrad.dtypes.int8)
	
	d = make_test_data(40)
	_test_function([d], {}, do_bitcast, do_bitcast)

def test_convert_fp16():
	for i, a in enumerate([make_test_data(1024).astype(np.float16), np.array([6.0975552e-5, 5.9604645e-8]).astype(np.float16), np.array([np.inf, np.inf]).astype(np.float16), np.array([np.nan, np.nan]).astype(np.float16) ]):
		a_t = tinygrad.Tensor(a.view(np.uint16) )
		out = tinybloat.compatibility.convert_fp16(a_t, tinygrad.dtypes.float32)
		error = mse(out.numpy(), a.astype(np.float32) )
		if i <2:
			assert error < 1.0e-4, f"error too high: {error}"
		else:
			if i == 3:
				# nan
				assert np.isnan(error)
			elif i == 2:
				# inf
				assert np.isinf(np.sum(out.numpy() ) )

def test_tensor_constructor():
	from tinygrad import dtypes, Tensor
	_test_dtypes = [
		(torch.float, dtypes.float),
		(torch.float16, dtypes.half),
		(torch.bfloat16, dtypes.bfloat16),
		(torch.float8_e4m3fn, dtypes.fp8e4m3),
		(torch.float8_e5m2, dtypes.fp8e5m2)
	]
	
	for torch_dt, tiny_dt in _test_dtypes:
		torch_tensor = torch.arange(16).to(torch.int32).view(torch_dt)
		tiny_tensor = tinybloat.tensor(np.arange(16).astype(np.int32), initial_dtype = tiny_dt)
		error = mse(torch_tensor.to(torch.float).numpy(), tiny_tensor.cast(dtypes.float).numpy() )
		assert error < 1.0e-4, f"error too large for {tiny_dt}: {error}"

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
			error = mse(qt.dequantize().numpy(), out)
			assert error < 5.0e-3, f"error too high: {error}"


