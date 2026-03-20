import numpy as np
import scipy.ndimage as ndimage

def calc_energy(im):
    filter_kernel = np.array([[0.5, 1.0, 0.5], [1.0, -6.0, 1.0], [0.5, 1.0, 0.5]])
    energy = np.zeros(im.shape[:2], dtype=np.float32)
    for c in range(3):
        res = ndimage.convolve(im[:, :, c].astype(np.float32), filter_kernel, mode='reflect')
        energy += res ** 2
    return energy

def reduce_width(im, delta):
    h, w = im.shape[:2]
    out = np.copy(im)
    for _ in range(delta):
        energy = calc_energy(out)
        M = np.copy(energy)
        backtrack = np.zeros_like(M, dtype=int)
        for i in range(1, h):
            for j in range(w):
                if j == 0:
                    idx = np.argmin(M[i-1, j:j+2])
                    backtrack[i, j] = idx + j
                    min_eng = M[i-1, idx + j]
                elif j == w - 1:
                    idx = np.argmin(M[i-1, j-1:j+1])
                    backtrack[i, j] = idx + j - 1
                    min_eng = M[i-1, idx + j - 1]
                else:
                    idx = np.argmin(M[i-1, j-1:j+2])
                    backtrack[i, j] = idx + j - 1
                    min_eng = M[i-1, idx + j - 1]
                M[i, j] += min_eng
        
        # find min at bottom
        j = np.argmin(M[-1])
        mask = np.ones((h, w), dtype=bool)
        for i in range(h-1, -1, -1):
            mask[i, j] = False
            j = backtrack[i, j]
        
        mask = np.stack([mask]*3, axis=2)
        out = out[mask].reshape((h, w-1, 3))
        w -= 1
    return out
