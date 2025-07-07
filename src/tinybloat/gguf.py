import gguf
import tinygrad
from pathlib import Path
from .quantization import QTensor

def gguf_load(fn: Union[tinygrad.Tensor, str, Path]) -> dict[str, tinygrad.Tensor]:
	"""
	TODO: Implement method that loads a GGUF file into a state dict.
	Must support quantized parameters.
	"""
	raise NotImplementedError
