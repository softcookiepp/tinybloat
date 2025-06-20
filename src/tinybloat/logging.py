WARNING_SUPPRESSION_MSG = ""

def warn_numpy_bandaid(function):
	print(f"Warning: Function `{function.__name__}` is currently implemented in numpy, and thus is neither differentiable or usable with tinygrad.TinyJit.", WARNING_SUPPRESSION_MSG)
