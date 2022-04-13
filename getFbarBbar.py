import numpy as np
from numba import jit

"""
This function solves for mean values of foreground and background
"""

@jit(nopython=True)
def getFbarBbar(foreground_kernel_color, background_kernel_color, w_f, w_b):

  weighted_f = w_f * foreground_kernel_color
  weighted_b = w_b * background_kernel_color
  W_f = np.sum(w_f[:, :, 1])
  W_b = np.sum(w_b[:, :, 1])

  f_bar1 = (1 / W_f) * np.sum(weighted_f[:, :, 0])
  f_bar2 = (1 / W_f) * np.sum(weighted_f[:, :, 1])
  f_bar3 = (1 / W_f) * np.sum(weighted_f[:, :, 2])
  f_bar = np.array([[f_bar1], [f_bar2], [f_bar3]])

  b_bar1 = (1 / W_b) * np.sum(weighted_b[:, :, 0])
  b_bar2 = (1 / W_b) * np.sum(weighted_b[:, :, 1])
  b_bar3 = (1 / W_b) * np.sum(weighted_b[:, :, 2])
  b_bar = np.array([[b_bar1], [b_bar2], [b_bar3]])

  return f_bar, b_bar