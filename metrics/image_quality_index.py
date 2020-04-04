
"""
Universal Image Quality Index (UIQI)

It's a measure of image similarity in the dynamic range of [-1; 1].

It's translated from MATLAB, found here:
https://github.com/plal/universal-img-quality-index/blob/master/img_qi_kernelized.m

MATLAB code:
------------

function quality = img_qi(img1, img2, kernelsize)

N = kernelsize.^2;
sum2_filter = ones(kernelsize);

img1 = double(img1);
img2 = double(img2);

img1_sq = img1.*img1;
img2_sq = img2.*img2;
img12 = img1.*img2;

img1_sum   = filter2(sum2_filter, img1, 'valid');
img2_sum   = filter2(sum2_filter, img2, 'valid');
img1_sq_sum = filter2(sum2_filter, img1_sq, 'valid');
img2_sq_sum = filter2(sum2_filter, img2_sq, 'valid');
img12_sum = filter2(sum2_filter, img12, 'valid');

img12_sum_mul = img1_sum.*img2_sum;
img12_sq_sum_mul = img1_sum.*img1_sum + img2_sum.*img2_sum;
top = 4*(N*img12_sum - img12_sum_mul).*img12_sum_mul;
bot = (N*(img1_sq_sum + img2_sq_sum) - img12_sq_sum_mul).*img12_sq_sum_mul;

quality = mean(mean(top./bot));

-----------------
(end MATLAB code)

The UIQI is a mean of Q scores for boxes of size BxB (with B == 8, conventionally)[1].

The formula is:

                4 * cov(x, y) * avg(x) * avg(y)
Q(x, y) = ---------------------------------------------
           (var(x) + var(y)) * (avg(x)**2 + avg(y)**2)


Using the so called shortcut formula for variance, and a similar for covariance,
we can get to the vectorized implementation below (N = B*B - number of elements):

                     4 * sum(x) * sum(y) * (N * sum(x * y) - sum(x) * sum(y))
Q(x, y) = -------------------------------------------------------------------------------------------
           (N * (sum(x**2)  + sum(y**2)) - sum(x) ** 2 - sum(y) ** 2) * (sum(x) ** 2  + sum(y) ** 2)

References:

[1] Wang, Zhou, and Alan C. Bovik. "A universal image quality index."
    IEEE signal processing letters 9.3 (2002): 81-84.
"""

import numpy as np
from scipy.signal import convolve2d


def universal_image_quality_index(x, y, kernelsize=8):
    """Compute the Universal Image Quality Index (UIQI) of x and y."""

    N = kernelsize ** 2
    kernel = np.ones((kernelsize, kernelsize))

    x = x.astype(np.float)
    y = y.astype(np.float)

    S_x = convolve2d(x, kernel, mode='valid')

    # to be continued...
