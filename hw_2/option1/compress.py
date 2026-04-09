import argparse
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from svd_impl import svd, reconstruct
from metrics import psnr, ssim


def load_gray(path):
    im = Image.open(path).convert('L')
    return np.asarray(im, dtype=float)


def save_gray(arr, path):
    arr = np.clip(arr, 0, 255).astype('uint8')
    Image.fromarray(arr).save(path)


def run(input_path, ks, outdir):
    os.makedirs(outdir, exist_ok=True)
    A = load_gray(input_path)
    m, n = A.shape
    U, S, Vt = svd(A)
    out_images = []
    metrics = []
    for k in ks:
        B = reconstruct(U, S, Vt, k)
        out_path = os.path.join(outdir, f'recon_k{k}.png')
        save_gray(B, out_path)
        out_images.append((k, out_path))
        metrics.append((k, psnr(A, B), ssim(A, B)))
    # make a summary plot
    cols = len(out_images) + 1
    fig, axes = plt.subplots(1, cols, figsize=(4*cols,4))
    axes[0].imshow(A, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    for i,(k,p) in enumerate(out_images, start=1):
        img = Image.open(p)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'k={k}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'summary.png'))
    # print metrics
    print('k	PSNR	SSIM')
    for k, p, s in metrics:
        print(f'{k}\t{p:.3f}\t{s:.4f}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--k', type=int, nargs='+', required=True)
    p.add_argument('--outdir', default='out')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args.input, args.k, args.outdir)
