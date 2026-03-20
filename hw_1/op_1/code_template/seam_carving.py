import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from skimage import io

## read image
im = io.imread('hw_1/op_1/figs/original.png')
if im.ndim == 3 and im.shape[2] == 4:
    im = im[:, :, :3]

## draw 2 copies of the image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.subplots_adjust(bottom=0.22)
ax1.imshow(im)
ax1.set_title('Input image')
ax1.axis('off')
himg = ax2.imshow(np.zeros_like(im))
ax2.set_title('Resized Image\nAdjust sliders and click the button')
ax2.axis('off')

slider_col_ax = fig.add_axes([0.15, 0.10, 0.30, 0.03])
slider_row_ax = fig.add_axes([0.15, 0.05, 0.30, 0.03])
slider_col = Slider(slider_col_ax, 'Col scale', 0.5, 2.0, valinit=1.0)
slider_row = Slider(slider_row_ax, 'Row scale', 0.5, 2.0, valinit=1.0)

btn_ax = fig.add_axes([0.60, 0.06, 0.20, 0.06])
btn = Button(btn_ax, 'Seam Carving', color='lightblue', hovercolor='deepskyblue')

def on_click(event):
    h, w = im.shape[:2]
    target_w = max(1, int(w * slider_col.val))
    target_h = max(1, int(h * slider_row.val))
    result = seam_carve_image(im, (target_h, target_w))
    himg.set_data(result)
    himg.set_extent([0, result.shape[1], result.shape[0], 0])
    ax2.set_title(f'Resized Image ({result.shape[0]}x{result.shape[1]})')
    fig.canvas.draw_idle()

btn.on_clicked(on_click)


from scipy import ndimage

def calc_energy(im):
    filter_kernel = np.array([[0.5, 1.0, 0.5], [1.0, -6.0, 1.0], [0.5, 1.0, 0.5]])
    energy = np.zeros(im.shape[:2], dtype=np.float32)
    for c in range(im.shape[2]):
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
        seam_idx = np.zeros(h, dtype=int)
        for i in range(h-1, -1, -1):
            seam_idx[i] = j
            j = backtrack[i, j]
        
        new_out = np.zeros((h, w+1, 3), dtype=out.dtype)
        for i in range(h):
            col = seam_idx[i]
            new_out[i, :col+1] = out[i, :col+1]
            if col < w - 1:
                new_out[i, col+1] = (out[i, col].astype(int) + out[i, col+1].astype(int)) // 2
            else:
                new_out[i, col+1] = out[i, col]
            if col + 1 < w:
                new_out[i, col+2:] = out[i, col+1:]
        out = new_out
    return out

def seam_carve_image(im, sz):
    """Seam carving to resize image to target size.

    Args:
        im: (h, w, 3) input RGB image (uint8)
        sz: (target_h, target_w) target size

    Returns:
        resized image of shape (target_h, target_w, 3)
    """
    target_h, target_w = sz
    h, w = im.shape[:2]
    out = np.copy(im)
    
    if target_w < w:
        out = reduce_width(out, w - target_w)
    elif target_w > w:
        out = expand_width(out, target_w - w)
        
    if target_h < h:
        out = np.transpose(out, (1, 0, 2))
        out = reduce_width(out, h - target_h)
        out = np.transpose(out, (1, 0, 2))
    elif target_h > h:
        out = np.transpose(out, (1, 0, 2))
        out = expand_width(out, target_h - h)
        out = np.transpose(out, (1, 0, 2))
        
    return out


plt.show()
