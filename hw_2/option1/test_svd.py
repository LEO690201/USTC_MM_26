import numpy as np
from svd_impl import svd, reconstruct


def rel_err(A, B):
    return np.linalg.norm(A - B) / (np.linalg.norm(A) + 1e-12)


def test_random(m, n):
    A = np.random.randn(m, n)
    U, S, Vt = svd(A)
    Ar = reconstruct(U, S, Vt, min(m,n))
    print('shape', A.shape, 'rel_err', rel_err(A, Ar))
    # compare singular values roughly with numpy
    s_np = np.linalg.svd(A, full_matrices=False, compute_uv=False)
    print('top singulars (ours) ', S[:5])
    print('top singulars (np)   ', s_np[:5])


if __name__ == '__main__':
    np.random.seed(0)
    test_random(8,5)
    test_random(5,8)
