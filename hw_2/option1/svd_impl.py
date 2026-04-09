import numpy as np

def svd(A, eps=1e-12):
    """
    Compute SVD A = U @ S @ Vt without calling high-level SVD.
    Uses eigen-decomposition of A^T A or A A^T depending on shape.
    Returns U (m x r), S (r,), Vt (r x n) where r = min(m,n)
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    if m >= n:
        # compute eigen-decomposition of A^T A (n x n)
        ATA = A.T @ A
        vals, V = np.linalg.eigh(ATA)
        # sort by descending eigenvalue
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        V = V[:, idx]
        # singular values
        s = np.sqrt(np.clip(vals, 0, None))
        # compute U = A V Σ^{-1}
        tol = eps
        r = min(m, n)
        U = np.zeros((m, r))
        S = np.zeros(r)
        Vt = V.T[:r]
        for i in range(r):
            sigma = s[i]
            if sigma > tol:
                ui = (A @ V[:, i]) / sigma
            else:
                ui = np.zeros(m)
            U[:, i] = ui
            S[i] = sigma
        return U, S, Vt
    else:
        # n > m: compute eigen-decomposition of A A^T (m x m)
        AAT = A @ A.T
        vals, U = np.linalg.eigh(AAT)
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        U = U[:, idx]
        s = np.sqrt(np.clip(vals, 0, None))
        r = min(m, n)
        Vt = np.zeros((r, n))
        S = np.zeros(r)
        tol = eps
        for i in range(r):
            sigma = s[i]
            if sigma > tol:
                vi = (A.T @ U[:, i]) / sigma
            else:
                vi = np.zeros(n)
            Vt[i, :] = vi
            S[i] = sigma
        return U[:, :r], S, Vt


def reconstruct(U, S, Vt, k):
    """Reconstruct using top-k singular values."""
    k = min(k, len(S))
    if k <= 0:
        return np.zeros((U.shape[0], Vt.shape[1]))
    Uk = U[:, :k]
    Sk = np.diag(S[:k])
    Vtk = Vt[:k, :]
    return Uk @ Sk @ Vtk

if __name__ == '__main__':
    # quick local test
    A = np.random.randn(8,5)
    U,S,Vt = svd(A)
    A2 = reconstruct(U,S,Vt,5)
    print('reconstruction error', np.linalg.norm(A-A2))
