import tinygrad
#from ..device import parse_device, get_default_device

def uniform_(tensor, a = 0.0, b = 1.0, generator = None):
	if not generator is None:
		raise NotImplementedError
	uni = tinygrad.Tensor.uniform(*tensor.shape, low = a, high = b,
		dtype = tensor.dtype, requires_grad = tensor.requires_grad,
		device = tensor.device).cast(tensor.dtype).to(tensor.device)
	return tensor.assign(uni)

uniform = uniform_

# hopefully this works
def normal_(tensor, mean = 0.0, std = 1.0, generator = None):
	if not generator is None:
		raise NotImplementedError
	norm = tinygrad.Tensor.normal(*tensor.shape,
		mean = mean,
		std = std,
		requires_grad = tensor.requires_grad,
		dtype = tensor.dtype,
		device = tensor.device).to(tensor.device)
	return tensor.assign(norm.cast(tensor.dtype))

normal = normal_

def trunc_normal_(tensor, mean = 0.0, std = 1.0, a = -2.0, b = 2.0, generator = None):
	if not generator is None:
		raise NotImplementedError
	norm = tinygrad.Tensor.normal(*tensor.shape,
		mean = mean,
		std = std,
		requires_grad = tensor.requires_grad,
		dtype = tensor.dtype,
		device = tensor.device).to(tensor.device)
	norm = norm.clamp(a, b)
	return tensor.assign(norm)

def constant_(tensor, val):
	full = tensor.full_like(val)
	return tensor.assign(full)
	

def xavier_uniform_(tensor, *args, **kwargs):
	new = tinygrad.Tensor.glorot_uniform(tensor.shape, device = tensor.device, dtype = tensor.dtype, requires_grad = tensor.requires_grad)
	return tensor.assign(new)

xavier_uniform = xavier_uniform_

def xavier_normal_(tensor, *args, **kwargs):
	raise NotImplementedError

xavier_normal = xavier_normal_

def kaiming_uniform_(tensor, a = 0.0, *args, **kwargs):
	norm = tinygrad.Tensor.kaiming_uniform(*tensor.shape,
		a = a, 
		dtype = tensor.dtype,
		device = tensor.device)
	return tensor.assign(norm.cast(tensor.dtype))

kaiming_uniform = kaiming_uniform_

def kaiming_normal_(tensor, *args, **kwargs):
	norm = tinygrad.Tensor.kaiming_normal(*tensor.shape,
		a = a, 
		dtype = tensor.dtype,
		device = tensor.device)
	return tensor.assign(norm.cast(tensor.dtype))

kaiming_normal = kaiming_normal_

def zeros_(tensor):
	return tensor.assign(tensor.zeros_like() )

def _calculate_correct_fan(*args, **kwargs):
	raise NotImplementedError
