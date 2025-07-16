import tinygrad
from typing import Union, Optional
import numpy as np
import inspect

class ComplexTensor:
	"""
	A tensor class that can be used for operations that require complex numbers.
	It stores the real and imaginary components as separate tinygrad tensors, but this
	may change if another means of storage is found to be more optimal.
	"""
	def __init__(self,
				real: Union[tinygrad.Tensor, np.ndarray],
				imag: Optional[Union[tinygrad.Tensor, np.ndarray]] = None, device = None, requires_grad = None):
		if isinstance(real, np.ndarray):
			if np.iscomplexobj(real):
				real, imag = real.real, real.imag
				imag = tinygrad.Tensor(imag, device = device, requires_grad = requires_grad)
			real = tinygrad.Tensor(real, device = device, requires_grad = requires_grad)
			if isinstance(imag, np.ndarray):
				imag = tinygrad.Tensor(imag, device = device, requires_grad = requires_grad)
		
		if imag is None:
			imag = real.zeros_like()
		assert isinstance(real, tinygrad.Tensor), f"Expected tinygrad.Tensor, got {type(real)} instead"
		if not device is None:
			real = real.to(device)
			imag = imag.to(device)
		self._real = real
		self._imag = imag
		assert self._real.device == self._imag.device
	
	@property
	def real(self) -> tinygrad.Tensor:
		"""
		The real component of the tensor
		"""
		return self._real
	
	@property
	def imag(self) -> tinygrad.Tensor:
		"""
		The imaginary component of the tensor
		"""
		return self._imag
	
	@property
	def device(self) -> str:
		"""
		The device that both the real and imaginary components are stored on.
		"""
		return self._real.device
		
	@property
	def ndim(self):
		return self._real.ndim
		
	@property
	def T(self):
		"""
		See [tinygrad.Tensor.T](https://docs.tinygrad.org/tensor/properties/#tinygrad.Tensor.T)
		"""
		return ComplexTensor(self._real.T, self._imag.T)
		
	@property
	def shape(self):
		"""
		See [tinygrad.Tensor.shape](https://docs.tinygrad.org/tensor/properties/#tinygrad.Tensor.shape)
		"""
		return self._real.shape
	
	@property
	def dtype(self):
		"""
		Unlike numpy or pytorch, tinygrad does not have complex64 and complex128 dtypes.
		Because of this, the underlying tinygrad dtype of the real component is returned instead.
		"""
		return self._real.dtype
		
	def to(self, *args, **kwargs):
		"""
		See [tinygrad.Tensor.to()](https://docs.tinygrad.org/tensor/properties/#tinygrad.Tensor.to)
		"""
		return self._tg_override(*args, **kwargs)
	
	def to_(self, device):
		"""
		See [tinygrad.Tensor.to_()](https://docs.tinygrad.org/tensor/properties/#tinygrad.Tensor.to_)
		"""
		self._real.to_(device)
		self._imag.to_(device)
		return self
	
	def cast(self, *args, **kwargs):
		"""
		See [tinygrad.Tensor.cast()](https://docs.tinygrad.org/tensor/properties/#tinygrad.Tensor.cast)
		"""
		return self._tg_override(*args, **kwargs)
	
	def numel(self):
		"""
		See [tinygrad.Tensor.numel()](https://docs.tinygrad.org/tensor/properties/#tinygrad.Tensor.numel)
		"""
		return self._real.numel()
	
	def replace(self, new):
		"""
		Replace this tensor with another.
		"""
		# lets be picky about it for now
		assert isinstance(new, ComplexTensor)
		self._real.replace(new.real)
		self._imag.replace(new.imag)
		return self

	def _tg_override(self, *args, **kwargs):
		# Method for automatically wrapping stuff coded in tinygrad so
		# stuff works correctly
		# This method will apply the given operation to both the real and
		# imaginary component, so use caution
		
		# this will be the function that gets wrapped
		tg_attr = inspect.stack()[1].function
		
		# ensure all tensors in args and kwar
		real_args = convert_to_real(args)
		real_kwargs = convert_to_real(kwargs)
		
		imag_args = convert_to_imag(args)
		imag_kwargs = convert_to_imag(kwargs)
		
		# lets just disable this for now...
		#assert_same_device(self.tg.device, tg_args, tg_kwargs)
		
		if len(real_kwargs) == 0:
			# fix for methods that don't support **kwargs
			out_real= self.real.__getattribute__(tg_attr)(*real_args)
			out_imag = self.imag.__getattribute__(tg_attr)(*imag_args)
		else:
			out_real = self.real.__getattribute__(tg_attr)(*real_args, **real_kwargs)
			out_imag = self.imag.__getattribute__(tg_attr)(*imag_args, **imag_kwargs)
		return ComplexTensor(out_real, out_imag)
	
	
	def contiguous(self):
		return self._tg_override()
	
	def realize(self):
		return self._tg_override()
	
	def reshape(self, *args, **kwargs):
		return self._tg_override(*args, **kwargs)
		
	def numpy(self):
		"""
		See [tinygrad.Tensor.numpy()](https://docs.tinygrad.org/tensor/properties/#tinygrad.Tensor.numpy)
		"""
		return self.real.numpy() + (1j* self.imag.numpy() )
	
	def __add__(self, other):
		a, b = self.real, self.imag
		if isinstance(other, tinygrad.Tensor):
			c, d = other, other.zeros_like()
		elif isinstance(other, ComplexTensor):
			c, d = other.real, other.imag
		else:
			# assume it is constant lol
			c, d = other, 0.0
		return ComplexTensor(a + c, b + d)
	
	def __sub__(self, other):
		return self._tg_override(other)
	
	def __rsub__(self, other):
		return self._tg_override(other)
		
	def __mul__(self, other):
		if isinstance(other, ComplexTensor):
			a, b = self.real, self.imag
			c, d = other.real, other.imag
			return ComplexTensor(a*c - b*d, a*d + b*c)
		else:
			return ComplexTensor(self.real * other, self.imag * other)
		
	def __div__(self, other):
		# complex division is a bit more silly
		if isinstance(other, ComplexTensor):
			# i totally forgot how to do this, so the logic is from here:
			# https://www.cuemath.com/numbers/division-of-complex-numbers/
			a, b = self.real, self.imag
			c, d = other.real, other.imag
			
			out_real = ( (a*c) + (b*d) ) / (c**2 + d**2)
			out_imag = ( (b*c) - (a*d) ) / (c**2 + d**2)
			return ComplexTensor(out_real, out_imag)
		else:
			# if complex component is 0, then regular division can be used
			return ComplexTensor(self.real/other, self.imag/other)
			
	def __truediv__(self, other):
		return self.__div__(other)
	
	def __rdiv__(self, other):
		# complex division is a bit more silly
		if isinstance(other, ComplexTensor):
			# i totally forgot how to do this, so the logic is from here:
			# https://www.cuemath.com/numbers/division-of-complex-numbers/
			a, b = other.real, other.imag
			c, d = self.real, self.imag
			
			out_real = ( (a*c) + (b*d) ) / (c**2 + d**2)
			out_imag = ( (b*c) - (a*d) ) / (c**2 + d**2)
			return ComplexTensor(out_real, out_imag)
		else:
			# still gotta do this crap
			a, b = other, 0
			c, d = self.real, self.imag
			
			out_real = ( (a*c) + (b*d) ) / (c**2 + d**2)
			out_imag = ( (b*c) - (a*d) ) / (c**2 + d**2)
			return ComplexTensor(out_real, out_imag)
	
	def __rtruediv__(self, other):
		return self.__rdiv__(other)
		
	def __matmul__(self, other):
		# oh god this will be painful
		assert len(other.shape) == 2
		
		other = convert_to_complex(other)
		
		# regular multiplication is
		# a, b = self.real, self.imag
		# c, d = other.real, other.imag
		# (a*c - b*d) + i*(a*d + b*c)
		
		# can we just do this?
		a, b = self.real, self.imag
		c, d = other.real, other.imag
		real = (a@c - b@d)
		imag = (a@d + b@c)
		# looks like it!
		return ComplexTensor(real, imag)
		
	def __rmatmul__(self, other):
		raise NotImplementedError
		
	def __getitem__(self, *args, **kwargs):
		return self._tg_override(*args, **kwargs)
	

def convert_to_complex(*inp):
	"""
	@private
	"""
	if len(inp) == 1:
		inp = inp[0]
	if isinstance(inp, ComplexTensor):
		return inp
	if isinstance(inp, tinygrad.Tensor):
		return ComplexTensor(inp)
	elif isinstance(inp, list) or isinstance(inp, tuple):
		new = []
		for item in inp:
			new.append(convert_to_complex(item) )
		if isinstance(inp, tuple):
			new = tuple(new)
		return new
	elif isinstance(inp, dict):
		for k, v in inp.items():
			inp[k] = convert_to_complex(v)
		return inp
	else:
		if hasattr(inp, "__dict__"):
			# treat as dictionary hehe
			new_dict = convert_to_complex(inp.__dict__)
			inp.__dict__.update(new_dict)
			return inp
		else:
			# inp is a primitive type
			return inp


def convert_to_real(*inp):
	"""
	@private
	"""
	if len(inp) == 1:
		inp = inp[0]
	if isinstance(inp, ComplexTensor):
		return inp.real
	if isinstance(inp, tinygrad.Tensor):
		return inp
	elif isinstance(inp, list) or isinstance(inp, tuple):
		new = []
		for item in inp:
			new.append(convert_to_real(item) )
		if isinstance(inp, tuple):
			new = tuple(new)
		return new
	elif isinstance(inp, dict):
		for k, v in inp.items():
			inp[k] = convert_to_real(v)
		return inp
	else:
		if hasattr(inp, "__dict__"):
			# treat as dictionary hehe
			new_dict = convert_to_real(inp.__dict__)
			inp.__dict__.update(new_dict)
			return inp
		else:
			# inp is a primitive type
			return inp

def convert_to_imag(*inp):
	"""
	@private
	Internal function
	"""
	if len(inp) == 1:
		inp = inp[0]
	if isinstance(inp, ComplexTensor):
		return inp.imag
	if isinstance(inp, tinygrad.Tensor):
		return inp.zeros_like()
	elif isinstance(inp, list) or isinstance(inp, tuple):
		new = []
		for item in inp:
			new.append(convert_to_imag(item) )
		if isinstance(inp, tuple):
			new = tuple(new)
		return new
	elif isinstance(inp, dict):
		for k, v in inp.items():
			inp[k] = convert_to_imag(v)
		return inp
	else:
		if hasattr(inp, "__dict__"):
			# treat as dictionary hehe
			new_dict = convert_to_imag(inp.__dict__)
			inp.__dict__.update(new_dict)
			return inp
		else:
			# inp is a primitive type
			return inp
