import numpy as np

"""Function to get kernel area"""

def getKernel(kernel_size, x_pos, y_pos, min_pix, Trimap):

  foreground_tri = (Trimap == 1)  # foreground alpha
  background_tri = (Trimap == 0)  # background alpha
  num_fore = 0
  num_back = 0
  M, N, Ch = np.shape(Trimap)

  while num_fore < min_pix or num_back < min_pix:
    half_size = np.fix(kernel_size / 2)
    ymin = int(max(0, y_pos - half_size))
    ymax = int(min(N, y_pos + half_size + 1))
    xmin = int(max(0, x_pos - half_size))
    xmax = int(min(M, x_pos + half_size + 1))
    num_fore = (foreground_tri[xmin:xmax, ymin:ymax, 1] == 1).sum()
    num_back = (background_tri[xmin:xmax, ymin:ymax, 1] == 1).sum()
    kernel_size = kernel_size + 4

  new_kernel_size = kernel_size - 4
  return xmin, xmax, ymin, ymax, new_kernel_size