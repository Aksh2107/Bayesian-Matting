import numpy as np

"""
This function solves for covariance matrices of foreground and background
"""

def getFcovBcov(foreground_kernel_color, background_kernel_color, f_bar, b_bar,
                w_f, w_b):

  f_weight = w_f.reshape((-1, 3))
  b_weight = w_b.reshape((-1, 3))

  f_pix = foreground_kernel_color.copy()
  f_pix = f_pix.reshape((-1, 3))
  f_pix = f_pix[f_weight > 0].reshape(-1, 3)
  f_weight = f_weight[f_weight > 0].reshape(-1, 3)
  f_pix = f_pix - np.transpose(f_bar)

  b_pix = background_kernel_color.copy()
  b_pix = b_pix.reshape((-1, 3))
  b_pix = b_pix[b_weight > 0].reshape(-1, 3)
  b_weight = b_weight[b_weight > 0].reshape(-1, 3)
  b_pix = b_pix - np.transpose(b_bar)

  b_pix = b_pix - np.transpose(b_bar)
  cov_f = np.cov(np.transpose(f_pix), bias=True,
                 aweights=np.transpose(f_weight[:, 1]))
  cov_b = np.cov(np.transpose(b_pix), bias=True,
                 aweights=np.transpose(b_weight[:, 1]))

  return cov_f, cov_b