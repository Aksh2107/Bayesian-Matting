import numpy as np
from numba import jit

"""
This function solves for F, B and a 
"""

@jit(nopython=True)
def matrixequation(C, alpha_bar, f_bar, b_bar, cov_f, cov_b, sigma_C, threshold,
                   max_it):

  it = 0.
  alpha = alpha_bar
  temp = 100000.
  while 1:

    I = np.eye(3)
    BR = np.linalg.pinv(cov_f) + (I * (alpha ** 2)) * (1 / sigma_C)
    BL = (I * alpha * (1 - alpha)) * (1 / sigma_C)
    TR = (I * alpha * (1 - alpha)) * (1 / sigma_C)
    TL = np.linalg.pinv(cov_b) + ((I * ((1 - alpha) ** 2)) * (1 / sigma_C))

    Down = np.dot(np.linalg.pinv(cov_f), f_bar) + (C * alpha / sigma_C)
    Up = np.dot(np.linalg.pinv(cov_b), b_bar) + (C * (1 - alpha) / sigma_C)

    big_matrix = np.vstack((np.hstack((TL, TR)), np.hstack((BL, BR))))
    small_matrix = np.vstack((Up, Down))

    FB = np.dot(np.linalg.pinv(big_matrix), small_matrix)
    F = FB[3:6]
    B = FB[0:3]
    d1 = C - B
    d2 = F - B
    new_alpha = np.sum(d1 * d2) / (np.linalg.norm(F - B) ** 2)
    alpha = max(0, min(1, new_alpha))

    if (it >= max_it) or (abs(alpha - temp) <= threshold):
      a = alpha
      break
    temp = alpha
    f_bar = F
    b_bar = B
    it = it + 1

  return F, B, a