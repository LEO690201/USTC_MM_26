import numpy as np


def psnr(img1, img2, max_val=255.0):
    img1 = np.asarray(img1, dtype=float)
    img2 = np.asarray(img2, dtype=float)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def ssim(img1, img2, max_val=255.0, K1=0.01, K2=0.03):
    """
    Full-image SSIM (single value) based on means/variances/covariance.
    This is a simplified, global SSIM (not sliding-window).
    """
    x = np.asarray(img1, dtype=float)
    y = np.asarray(img2, dtype=float)
    if x.shape != y.shape:
        raise ValueError('Input images must have the same shape')
    L = max_val
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x2 = ((x - mu_x) ** 2).mean()
    sigma_y2 = ((y - mu_y) ** 2).mean()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
    if den == 0:
        return 1.0
    return num / den


if __name__ == '__main__':
    import numpy as np
    a = np.arange(16).reshape(4,4).astype(float)
    b = a + np.random.randn(*a.shape)
    print('PSNR', psnr(a,b))
    print('SSIM', ssim(a,b))
