import numpy as np

"""
Function to mimic the 'fspecial' gaussian MATLAB function
"""

def kernelg(sigma, new_kernel_size, x_pos, y_pos, M, N):

  half_size = np.fix(new_kernel_size / 2)
  x, y = np.mgrid[-new_kernel_size // 2 + 1:new_kernel_size // 2 + 1, -
                  new_kernel_size // 2 + 1:new_kernel_size // 2 + 1]

  g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
  g / g.sum()

  if (new_kernel_size < 51):
    g = g / np.max(g)
  else:
    g = g / np.max(g)
    g = g + 0.001
    g = g / np.max(g)

  if ((x_pos - half_size) < 0):
    xxmin = int(0 - (x_pos - half_size + 1) + 1)
  else:
    xxmin = 0

  if ((x_pos + half_size) >= M):
    xxmax = int(new_kernel_size - (x_pos + half_size - M + 1))
  else:
    xxmax = new_kernel_size

  if(y_pos - half_size < 0):
    yymin = int(0 - (y_pos - half_size + 1) + 1)
  else:
    yymin = 0

  if ((y_pos + half_size) >= N):
    yymax = int(new_kernel_size - (y_pos + half_size - N + 1))
  else:
    yymax = new_kernel_size
  g = g[xxmin:xxmax, yymin: yymax]

  return g