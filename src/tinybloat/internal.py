def get_tensor_memoryview(t):
	return t.realize().uop.base.buffer.copyout(memoryview( bytearray(t.nbytes() ) ) )
	
