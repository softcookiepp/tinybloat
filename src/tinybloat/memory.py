import tinygrad

def free_all_buffers():
	# useful if freeing everything is required
	for k, v in tinygrad.ops.buffers.items():
		v.deallocate()
		del tinygrad.ops.buffers[k]
