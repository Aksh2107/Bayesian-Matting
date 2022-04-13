import cv2
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from getKernel import getKernel
from kernelg import kernelg
from getFbarBbar import getFbarBbar
from getFcovBcov import getFcovBcov
from matrixequation import matrixequation


def BayesianMatting(kernel_size, min_pix, sigma, threshold, max_it, Ground_Truth
                    , Original_Image, Trimap):

  # Convert to numpy formart
  Ground_Truth = np.asarray(Ground_Truth)
  Original_Image = np.asarray(Original_Image)
  Trimap = np.asarray(Trimap)

  M, N, channels = Original_Image.shape  # get the size of input image

  #normalise
  Ground_Truth = Ground_Truth / 255  # Ground Truth
  Original_Image = Original_Image / 255  # Original image
  Trimap = Trimap / 255  # Trimap

  # calculate the variance of the input image
  sigma_C = np.std(Original_Image) ** 2

  # get the definite foreground and background and unknown region
  foreground_tri = (Trimap == 1)  # definite foreground alpha
  background_tri = (Trimap == 0)  # definite background alpha
  Onesmatrix = np.ones((M, N, 3))
  # get definite unknown alpha
  unknown_tri = Onesmatrix - background_tri - foreground_tri
  unknown_tri_ind = np.zeros((M, N, 3))  # alpha indicator
  unknown_tri_it = np.zeros((M, N, 3))
  # get definite foreground and background
  foreground_color = Original_Image * foreground_tri
  background_color = Original_Image * background_tri

  # get the positions of unknown pixels
  unpos = np.argwhere(unknown_tri[:, :, :1] == 1)
  x = unpos[:, 0]  # rows
  y = unpos[:, 1]  # columns

  for val in tqdm(range(np.size(x))):
    x_pos = x[val]
    y_pos = y[val]

    #get the kernel area around each unknown pixel
    xmin, xmax, ymin, ymax, new_kernel_size = getKernel(kernel_size, x_pos,
                                                        y_pos, min_pix, Trimap)

    # get definite foreground and background in the kernel
    foreground_kernel_color = foreground_color[xmin:xmax, ymin:ymax, :]
    background_kernel_color = background_color[xmin:xmax, ymin:ymax, :]
    unknown_tri_it_kernel = unknown_tri_it[xmin:xmax, ymin:ymax]
    foreground_kernel_tri = foreground_tri[xmin:xmax, ymin:ymax]
    background_kernel_tri = background_tri[xmin:xmax, ymin:ymax]
    unknown_tri_ind_kernel = unknown_tri_ind[xmin:xmax, ymin:ymax]

    # get opacity of the kernel
    kernel_forealpha_with_un = foreground_kernel_tri + unknown_tri_it_kernel
    foreground_opacity = kernel_forealpha_with_un ** 2
    kernel_backalpha_with_un = ((background_kernel_tri + unknown_tri_ind_kernel)
    * ((np.ones((np.shape(unknown_tri_it_kernel)))) - kernel_forealpha_with_un))
    background_opacity = kernel_backalpha_with_un ** 2

    # get the Gaussian fall off of the kernel
    g = kernelg(sigma, new_kernel_size, x_pos, y_pos, M, N)
    g = g / np.max(g)

    #calculate the foreground and background weight
    w_f = foreground_opacity * g[:, :, None]
    w_b = background_opacity * g[:, :, None]

    # calculate the mean of the foreground and background pixels in the kernel
    f_bar, b_bar = getFbarBbar(foreground_kernel_color, background_kernel_color,
                               w_f, w_b)

    # calculate the covariance matrices
    cov_f, cov_b = getFcovBcov(foreground_kernel_color, background_kernel_color,
                               f_bar, b_bar, w_f, w_b)

    #calculate mean alpha value in the kernel
    total_num_alpha = np.sum(background_kernel_tri + foreground_kernel_tri +
                             unknown_tri_ind_kernel)
    alpha_bar = np.sum(kernel_forealpha_with_un) / total_num_alpha

    C = Original_Image[x_pos, y_pos, :].reshape(3, 1)

    #slove for F, B and alpha
    F, B, a = matrixequation(C, alpha_bar, f_bar, b_bar, cov_f, cov_b, sigma_C,
                             threshold, max_it)

    #reuse the calculated F, B and a
    unknown_tri_ind[x_pos, y_pos, :] = 1
    unknown_tri_it[x_pos, y_pos, :] = a
    foreground_color[x_pos, y_pos, :] = F.reshape((1, 1, 3))
    background_color[x_pos, y_pos, :] = B.reshape((1, 1, 3))


  new_alpha_map = unknown_tri_it + foreground_tri

  #output images
  alpha_fore = foreground_tri + unknown_tri_it
  new_fore = foreground_color * alpha_fore
  final_im = foreground_color * alpha_fore + (1 - alpha_fore) * background_color
  diff = abs(new_alpha_map - Ground_Truth)
  im_diff = abs(final_im - Original_Image)

  #output matrics
  MSE = mean_squared_error(Original_Image * 255, final_im * 255)
  MSE_alpha = mean_squared_error(new_alpha_map * 255, Ground_Truth * 255)
  estimated_PSNR = cv2.PSNR(Original_Image * 255, final_im * 255)
  Ground_Truth_float32 = np.float32(Ground_Truth)
  new_alpha_map_float32 = np.float32(new_alpha_map)
  Ground_Truth_lum = cv2.cvtColor(Ground_Truth_float32 * 255, cv2.COLOR_BGR2GRAY)
  new_alpha_lum = cv2.cvtColor(new_alpha_map_float32 * 255, cv2.COLOR_BGR2GRAY)
  sim = ssim(Ground_Truth[:, :, 1], new_alpha_map[:, :, 1])
  sim2 = ssim(Ground_Truth_lum, new_alpha_lum)

  print('The MSE between the output alpha map and the ground truth:')
  print('MSE(alpha)=', round(MSE_alpha, 4))
  print('The MSE between the original image and the estimated image:')
  print('MSE(image)=', round(MSE, 4))
  print('The PSNR between the output original image and the estimated image:')
  print('PSNR(image)=', round(estimated_PSNR, 4), 'dB')
  print('The SSIM between the output alpha map and the ground truth:')
  print('SSIM(alpha)=', round(sim, 4))
  print('The SSIM between the output original image and the estimated image:')
  print('SSIM(image)=', round(sim2, 4))

  return new_alpha_map, diff, final_im, im_diff, new_fore