import cv2
import sys
from BayesianMatting import BayesianMatting
from ShowNwriteIm import ShowNwriteIm


if __name__ == '__main__':

  #User input
  Original_im_path = input("Please input the file name of the Original Image: ")
  if (Original_im_path.endswith(('.png', '.jpg', '.jpeg'))):
    print('Original Image', Original_im_path,  'Input Successful!')
  else:
    print('Please input an .png or .jpg or .jpeg file')
    sys.exit()
  Tri_path = input("Please input the file name of the Trimap: ")
  if (Tri_path.endswith(('.png', '.jpg', '.jpeg'))):
    print('Trimap', Tri_path, 'Input Successful!')
  else:
    print('Please input an .png or .jpg or .jpeg file')
    sys.exit()
  GT_path = input("Please input the file name of the Ground Truth: ")
  if (GT_path.endswith(('.png', '.jpg', '.jpeg'))):
    print('Ground Truth', GT_path, 'Input Successful!')
  else:
    print('Please input an .png or .jpg or .jpeg image')
    sys.exit()

  #set input path
  Original_im_path = "inputs/" + Original_im_path
  Tri_path = "inputs/" + Tri_path
  GT_path = "inputs/" + GT_path

  #set parameters
  kernel_size = 31  # initial kernel size (odd)
  min_pix = 150  # minimum required pixels in a kernel
  sigma = 8  # variance of Gaussian mask
  threshold = 1e-5  # stopping threshold
  max_it = 150  # maximum iterations

  #read images
  Ground_Truth = cv2.imread(GT_path)[:, :, :3]
  Original_Image = cv2.imread( Original_im_path)[:, :, :3]
  Trimap = cv2.imread(Tri_path)[:, :, :3]

  #Bayesian Matting
  new_alpha_map, diff, final_im, im_diff, new_fore = \
  BayesianMatting(kernel_size, min_pix, sigma, threshold, max_it, Ground_Truth,
                  Original_Image, Trimap)
  #outputs
  ShowNwriteIm(new_alpha_map, diff, final_im, im_diff, new_fore)







