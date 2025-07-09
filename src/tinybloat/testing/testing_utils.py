from tinygrad.nn.state import torch_load, safe_load, load_state_dict, get_state_dict
import safetensors
from safetensors.torch import save_file
import tinygrad
import os
import numpy as np
import tinybloat


_has_pil = False
try:
	from PIL import Image
except ImportError:
	# maybe print a message here to tell testers they can install PIL to see image outputs?
	_has_pil = False
	
_has_tg_adapter = False
try:
	import tg_adapter
	tga = tg_adapter
except ImportError:
	# whatever, who cares
	_has_tg_adapter = False
	
try:
	import torch
except ImportError:
	raise ImportError("Pytorch is required to run the testing suite (GPU capability not required.)\nPlease install it from https://pytorch.org/get-started/locally/")

def _get_attributes_from_key(torch_root, tg_root, sd_key):
	attr_chain = sd_key.split(".")
	tg_obj = tg_root
	torch_obj = torch_root
	keys = []
	for k in attr_chain:
		keys.append(k)
		try:
			i = int(k)
			torch_obj = torch_obj[i]
			tg_obj = tg_obj[i]
			
		except ValueError:
			# not an int
			try:
				torch_obj.__getattribute__(k)
				torch_obj = torch_obj.__getattribute__(k)
			except AttributeError:
				torch_obj = torch_obj.__getattr__(k)
			
			try:
				tg_obj.__getattribute__(k)
				tg_obj = tg_obj.__getattribute__(k)
			except AttributeError:
				tg_obj = tg_obj.__getattr__(k)
			
		yield ".".join(keys), torch_obj, tg_obj

def get_submodules(torch_module, tg_module):
	# first, get all the state dict keys
	tg_sd = tg_module.state_dict()
	torch_sd = torch_module.state_dict()
	
	
	tg_submodules = {}
	torch_submodules = {}
	
	# then, determine which ones are modules
	for sd_key in tg_sd.keys():
		for k, torch_v, tg_v in _get_attributes_from_key(torch_module, tg_module, sd_key):
			if (not k in tg_submodules.keys() ) and ( isinstance(torch_v, torch.nn.Module) ):
				# key is all good, lets add it
				tg_submodules[k] = tg_v
				torch_submodules[k] = torch_v
	assert len(torch_submodules) == len(tg_submodules)
	return torch_submodules, tg_submodules

def _make_input_tensor(arg):
	return np.random.randn(int(np.prod(arg.shape)) ).astype(arg.numpy().dtype).reshape(arg.shape)
	
def _test_submodule(torch_module, tg_module):
	# might as well
	compare_state_dicts(torch_module, tg_module)
	copy_state_dict(torch_module, tg_module)
	args = []
	for arg in tg_module._input_spec[0]:
		args.append(_process_submodule_test_arg(arg) )
	kwargs = {}
	for k, arg in tg_module._input_spec[1].items():
		kwargs[k] = _process_submodule_test_arg(arg)
	_test_hf_reimplementation(args, kwargs, torch_module, "__call__", tg_module, "__call__")
	
def _test_all_submodules(torch_module, tg_module):
	torch_submodules, tg_submodules = get_submodules(torch_module, tg_module)
	for k in torch_submodules.keys():
		torch_sub = torch_submodules[k]
		tg_sub = tg_submodules[k]
		if isinstance(tg_sub, list):
			# module lists cannot be called
			continue
		
		# must have input spec in order to run any tests whatsoever
		if not tg_sub._input_spec is None:
			_test_submodule(torch_sub, tg_sub)

def compare_state_dicts(torch_module, tga_module, error_threshold = 1.0e-9):
	#print(type(torch_module), type(tga_module) )
	torch_sd = torch_module.state_dict()
	try:
		tga_sd = tga_module.state_dict()
	except AttributeError:
		tga_sd = get_state_dict(tga_module)
	#print(torch_sd.keys() )
	#print(tga_sd.keys() )
	try:
		if len(tga_sd.keys() ) != len(torch_sd.keys() ):
			raise ValueError
		for torch_key, tga_key in zip(sorted(torch_sd.keys() ), sorted(tga_sd.keys() ) ):
			#print(torch_key, tga_key)
			if torch_key != tga_key.replace("._tg", ""):
				print("Key mismatch:", torch_key)
			key = torch_key
			torch_value = torch_sd[key].detach().numpy()
			tga_value = tga_sd[tga_key]._tg.to("CPU").realize().numpy()
			
			error = mse(torch_value, tga_value)
			if error >= error_threshold or np.isnan(error):
				print("state dict values don't match for", torch_key)
			assert error < error_threshold
	except AssertionError:
		# keys are not equal
		tga_sd_norm = []
		for k in tga_sd.keys():
			tga_sd_norm.append(k.replace("._tg", "") )
		
		# keys that tga module doesn't have
		missing_keys_tg = list(set(torch_sd.keys() ) - set(tga_sd_norm ) )

		missing_keys_torch = list( set(tga_sd_norm ) - set(torch_sd.keys() ) )
		
		print("keys missing from tga module:")
		print(missing_keys_tg)
		print(f"\n{len(missing_keys_torch)} keys missing from torch module:")
		print(missing_keys_torch)
		input()
		raise ValueError


def inspect_state_dict_devices(module):
	for k, v in module.state_dict().items():
		print(k, v.device)
	input("meep")

def copy_state_dict(torch_module, tga_module, use_tg_adapter = _has_tg_adapter):
	# this should work
	d = os.path.abspath(".")
	if os.path.exists("/dev/shm"):
		d = "/dev/shm"
	fn = os.path.join(d, "tmp.safetensors")
	save_file(torch_module.state_dict(), fn)
	state_dict = safe_load(fn)
	if use_tg_adapter:
		tga_module.load_state_dict(state_dict)
		compare_state_dicts(torch_module, tga_module)
	else:
		load_state_dict(tga_module, state_dict)
	os.remove(fn)
	
def normalize(v):
	norm = np.linalg.norm(v)
	if norm == 0: 
		return v
	return v / norm

def norm_mse(predicted, actual):
	return mse( normalize(predicted), normalize(actual) )

def mse(predicted, actual):
	assert predicted.shape == actual.shape, f"Shapes differ: {actual.shape} and {predicted.shape}"
	return (np.sum( (predicted - actual)**2 ))/predicted.size
	
def make_test_data(*shape):
	return np.random.randn(int(np.prod(shape)) ).reshape(shape).astype(np.float32)
	
def _test_function(inp_args, inp_kwargs, torch_function, tinygrad_function, error_threshold = 1.0e-7):
	_test_hf_reimplementation( inp_args, inp_kwargs, torch_function, "__call__", tinygrad_function, "__call__", error_threshold = error_threshold)
	

def _print_dict_types(d):
	for k, v in d.items():
		print(k, type(v), v)
	
def str_to_numerical(s: str):
	b = bytes(s, "utf-8")
	return np.array(memoryview(b) ).astype(np.float32)

def _test_key_errors(hf_dict, tg_dict, error_threshold = 1.0e-6, print_values = True, display_images = False, error_function = mse):
	print("types:", type(hf_dict), type(tg_dict) )
	if isinstance(hf_dict, dict):
		tested_keys = hf_dict.keys()
		if list(hf_dict.keys() ) != list(tg_dict.keys() ):
			print("Warning! keys don't match! Will only test the ones that do")
			print(hf_dict.keys() )
			print(tg_dict.keys() )
			if len(hf_dict.keys() ) > len(tg_dict.keys() ):
				tested_keys = tg_dict.keys()
		for k in tested_keys:
			print("key:", k)
			hf_item = hf_dict[k]
			tg_item = tg_dict[k]
			_test_key_errors(hf_item, tg_item, error_threshold, print_values, display_images, error_function)
	elif isinstance(hf_dict, torch.Tensor):
		_test_key_errors(hf_dict.detach().numpy(), tg_dict.numpy(), error_threshold, display_images, error_function)
	elif isinstance(hf_dict, Image.Image):
		hf_dict.save("test_hf.png")
		tg_dict.save("test_tg.png")
		hf_item, tg_item = np.array(hf_dict), np.array(tg_dict)
		_test_key_errors(hf_item, tg_item, error_threshold, display_images, error_function)
	elif isinstance(hf_dict, list):
		if isinstance(tg_dict, np.ndarray):
			hf_dict = np.array(hf_dict).astype(np.float32)
			_test_key_errors(hf_dict, tg_dict, error_threshold, display_images, error_function)
		else:
			if len(hf_dict) != len(tg_dict):
				print("Lengths differ!")
				hf_dict = set(hf_dict)
				tg_dict = set(tg_dict)
				if len(tg_dict) < len(hf_dict):
					print("Values missing from tg:", hf_dict - tg_dict)
				raise ValueError
			# list of other sort, non-numerical
			for hf_item2, tg_item2 in zip(hf_dict, tg_dict):
				_test_key_errors(hf_item2, tg_item2, error_threshold, display_images, error_function)
	
	elif isinstance(hf_dict, tuple):
		# tuple of crap
		for hf_item2, tg_item2 in zip(hf_dict, tg_dict):
			_test_key_errors(hf_item2, tg_item2, error_threshold, display_images, error_function)
	elif isinstance(hf_dict, object) and hasattr(hf_dict, "__dict__"):
		_test_key_errors(hf_dict.__dict__, tg_dict.__dict__, error_threshold, display_images, error_function)
	elif isinstance(hf_dict, np.ndarray):
		error = mse(hf_dict, tg_dict)
		print("value mse:", error, "\n")
		if error > error_threshold or np.isnan(error):
			print(hf_dict.shape, tg_dict.shape)
			print(hf_dict)
			print(tg_dict)
			input()
	elif type(hf_dict) in [int, float]:
		_test_key_errors(np.array(hf_dict), np.array(tg_dict), error_threshold, display_images, error_function)
	elif isinstance(hf_dict, str):
		hf_dict = str_to_numerical(hf_dict)
		tg_dict = str_to_numerical(tg_dict)
		_test_key_errors(hf_dict, tg_dict, error_threshold, display_images, error_function)
	elif isinstance(hf_dict, type(None) ):
		pass
	elif isinstance(hf_dict, bool):
		_test_key_errors(int(hf_dict), int(tg_dict), error_threshold, display_images, error_function)
	elif hasattr(hf_dict, "__iter__"):
		assert len(hf_dict) == len(tg_dict), f"got differing lengths: {len(hf_dict)} and {len(tg_dict)}"
		for hf_item, tg_item in zip(sorted(hf_dict), sorted(tg_dict) ):
			_test_key_errors(hf_item, tg_item, error_threshold, display_images, error_function)
	else:
		raise ValueError
	
def _process_arg(arg, device, use_tg_adapter):
	if isinstance(arg, np.ndarray):
		# convert to tensor
		if use_tg_adapter:
			tgt = tg_adapter.tensor(arg).to(device)
			tgt.tg.realize()
		else:
			if np.iscomplexobj(arg):
				tgt = tinybloat.ComplexTensor(arg).to(device).realize()
			else:
				tgt = tinygrad.Tensor(arg).to(device).realize()
		return torch.tensor(arg), tgt
	elif isinstance(arg, list):
		hf_list, tg_list = [], []
		for arg2 in arg:
			h, t = _process_arg(arg2, device, use_tg_adapter)
			hf_list.append(h)
			tg_list.append(t)
		return hf_list, tg_list
	elif isinstance(arg, tuple):
		alist = list(arg)
		hf_list, tg_list = _process_arg(alist, device, use_tg_adapter)
		return tuple(hf_list), tuple(tg_list)
	else:
		# append as is
		return arg, arg
		
def _process_submodule_test_arg(arg):
	if isinstance(arg, tg_adapter.Tensor):
		# convert to tensor
		return _make_input_tensor(arg)
	elif isinstance(arg, list):
		arg_list = []
		for arg2 in arg:
			a = _process_submodule_test_arg(arg2)
			arg_list.append(a)
		return arg_list
	elif isinstance(arg, tuple):
		alist = list(arg)
		arg_list = _process_submodule_test_arg(alist)
		return tuple(arg_list)
	else:
		# append as is
		return arg

def _test_hf_reimplementation(args, kwargs, hf_module, hf_method, my_module, my_method, error_threshold = 1.0e-5, move_after_creation = True, device = "cuda:0", display_images = False, use_tg_adapter = False):
	if not (isinstance(args, tuple) or isinstance(args, list) ):
		args = (args,)
	if hasattr(my_module, "to") and move_after_creation and use_tg_adapter:
		my_module = my_module.to(device)
	elif (not use_tg_adapter):
		# just use default tinygrad device
		# still need to write a function that lets you move a module to device
		device = tinygrad.Device.DEFAULT
		tinybloat.move_to_device(my_module, device)
	hf_args, my_args = [], []
	for arg in args:
		torch_v, tg_v = _process_arg(arg, device, use_tg_adapter)
		hf_args.append(torch_v)
		my_args.append(tg_v )
	
	hf_args = tuple(hf_args)
	my_args = tuple(my_args)
	
	hf_kwargs = {}
	my_kwargs = {}
	for k, v in kwargs.items():
		torch_v, tg_v = _process_arg(v, device, use_tg_adapter)
		hf_kwargs[k] = torch_v
		my_kwargs[k] = tg_v
	
	
	
	# compute tiny out first so we don't have to wait for torch
	if isinstance(hf_method, str):
		tiny_out = my_module.__getattribute__(my_method)(*my_args, **my_kwargs)
		torch_out = hf_module.__getattribute__(hf_method)(*hf_args, **hf_kwargs)
	else:
		# function substitute
		if use_tg_adapter:
			tiny_out = my_method(my_module, tg_adapter, *my_args, **my_kwargs)
		else:
			tiny_out = my_method(my_module, tinygrad, *my_args, **my_kwargs)
		torch_out = hf_method(hf_module, torch, *hf_args, **hf_kwargs)
	
	print(f"MSE for {hf_module} and {my_module}:")
	_test_key_errors(torch_out, tiny_out, display_images = display_images, error_threshold = error_threshold, error_function = mse)
	
	del torch_out
	del tiny_out
