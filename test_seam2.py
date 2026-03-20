import numpy as np
import scipy.ndimage as ndimage

def calc_energy(im):
    filter_kernel = np.array([[0.5, 1.0, 0.5], [1.0, -6.0, 1.0], [0.5, 1.0, 0.5]])
    energy = np.zeros(im.shape[:2], dtype=np.float32)
    for c in range(3):
        res = ndimage.convolve(im[:, :, c].astype(np.float32), filter_kernel, mode='reflect')
        energy += np.square(res)
    return energy

def reduce_width(im, delta):
    out = np.copy(im)
    for _ in range(delta):
        h, w = out.shape[:2]
        energy = calc_energy(out)
        M = np.copy(energy)
        backtrack = np.zeros_like(M, dtype=int)
        for i in range(1, h):
            M_prev = M[i-1]
            M_prev_padded = np.pad(M_prev, (1, 1), constant_values=np.inf)
            idx = np.argmin([M_prev_padded[:-2], M_prev_padded[1:-1], M_prev_padded[2:]], axis=0)
            backtrack[i] = idx - 1 + np.arange(w)
            M[i] += np.min([M_prev_padded[:-2], M_prev_padded[1:-1], M_prev_padded[2:]], axis=0)
        
        j = np.argmin(M[-1])
        mask = np.ones((h, w), dtype=bool)
        for i in range(h-1, -1, -1):
            mask[i, j] = False
            j = backtrack[i, j]
        
        mask = np.stack([mask]*3, axis=2)
        out = out[mask].reshape((h, w-1, 3))
    return out

def expand_width(im, delta):
    out = np.copy(im)
    # Fast approach: duplicate `delta` lowest energy seams in order.
    # To prevent finding the same seam, we add a very high energy to the seam.
    h, w = out.shape[:2]
    seams = []
    
    # Need to find delta seams on the original image
    temp_im = np.copy(im)
    for _ in range(delta):
        energy = calc_energy(temp_im)
        M = np.copy(energy)
        backtrack = np.zeros_like(M, dtype=int)
        for i in range(1, h):
            M_prev = M[i-1]
            M_prev_padded = np.pad(M_prev, (1, 1), constant_values=np.inf)
            idx = np.argmin([M_prev_padded[:-2], M_prev_padded[1:-1], M_prev_padded[2:]], axis=0)
            backtrack[i] = idx - 1 + np.arange(temp_im.shape[1])
            M[i] += np.min([M_prev_padded[:-2], M_prev_padded[1:-1], M_prev_padded[2:]], axis=0)
            
        j = np.argmin(M[-1])
        seam = []
        mask = np.ones((h, temp_im.shape[1]), dtype=bool)
        for i in range(h-1, -1, -1):
            seam.append(j)
            mask[i, j] = False
            j = backtrack[i, j]
        seam.reverse()
        seams.append(seam)
        
        mask3 = np.stack([mask]*3, axis=2)
        temp_im = temp_im[mask3].reshape((h, temp_im.shape[1]-1, 3))
    
    # We found `delta` seams by removing them, which means their coordinates correspond to 
    # consecutive reduced images.  Instead of mapping coordinates, a simpler heuristic for 
    # expanding by delta (where delta < w) is to compute the DP on original image, find min seam, duplicate it it in out,
    # and in the next DP, to avoid duplicating it again, we just do it iteratively on 'out' but that leads to artificial effects.
    # Actually, iterative insertion is OK for simple assignments.
    pass
