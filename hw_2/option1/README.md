# HW2 - Option 1: SVD Image Compression (Python)

This folder contains a minimal implementation of SVD (without calling high-level SVD library functions) and a demo to compress grayscale images by keeping the top-k singular values.

Files:
- `svd_impl.py`: hand-crafted SVD using eigen-decomposition of A^T A or A A^T.
- `metrics.py`: PSNR and a full-image SSIM implementation.
- `compress.py`: demo script to compress an input image at specified ranks and save outputs.
- `test_svd.py`: small test comparing `svd_impl.svd` against `numpy.linalg.svd` on random matrices.
- `requirements.txt`: Python dependencies.

Usage example:
```
python3 compress.py --input path/to/image.jpg --k 5 20 50 --outdir out
```
