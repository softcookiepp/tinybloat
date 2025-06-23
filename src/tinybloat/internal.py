def get_tensor_memoryview(t):
	return t.uop.base.buffer.copyout(memoryview( bytearray(t.nbytes() ) ) )
	
