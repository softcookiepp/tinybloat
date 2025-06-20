import tinygrad
import numpy as np
from .logging import warn_numpy_bandaid
from .common import diag
from .complex_tensor import ComplexTensor

def norm(A, ord = None, dim = None, keepdim = False, out = None, dtype = None):
	if (not ord is None) and ord != 2:
		raise NotImplementedError(f"Order not implemented for linalg.norm: {ord}")
	return ( (A**2).sum(axis = dim, keepdim = keepdim) )**0.5

def qr(A, mode = "reduced"):
	if not mode is "reduced":
		raise NotImplementedError(f"mode not implemented for tg_adapter.linalg.qr: {mode}")
	if True:
		warn_numpy_bandaid(qr)
		Anp = A.numpy()
		QR = np.linalg.qr(Anp)
		Q = tinygrad.Tensor(QR.Q.astype(Anp.dtype), device = A.device)
		R = tinygrad.Tensor(QR.R.astype(Anp.dtype), device = A.device)
		return Q, R
	else:
		if len(A.shape) > 2:
			raise ValueError(f"Expected A to be 1D or 2D, but got {len(A.shape)}D instead")
		elif len(A.shape) == 1:
			A = A.reshape(1, -1)
			
		m, n = A.shape
		
		
		R = A.zeros_like().contiguous()
		R[:] = A[:]
		
		Q = tinygrad.Tensor.eye(m, dtype = A.dtype, device = A.device)
		
		for j in range(0, n):
			v, tau = _householder_vectorized(R[j:, j].reshape(-1, 1))
			H = tinygrad.Tensor.eye(m, dtype = A.dtype, device = A.device).contiguous()
			H[j:, j:] -= tau * (v @ v.T)
			R = H @ R
			Q = H @ Q
		return Q[:n].T, R[:n].triu()

def eig(A, max_iter = 100, tol = 1e-6):
	Anp = A.numpy()
	result = np.linalg.eig(Anp)
	eigenvalues = ComplexTensor(result.eigenvalues, device = A.device)
	eigenvectors = ComplexTensor(result.eigenvectors, device = A.device)
	
	return eigenvalues, eigenvectors
