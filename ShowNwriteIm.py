import cv2


def ShowNwriteIm(new_alpha_map, diff, final_im, im_diff, new_fore):

  """
  Write images
  """
  cv2.imwrite("outputs/new_alpha_map.png", new_alpha_map * 255)
  cv2.imwrite("outputs/alpha_difference.png", diff * 255)
  cv2.imwrite("outputs/estimated_image.png", final_im * 255)
  cv2.imwrite("outputs/difference_between_estimated_image_and_origninal.png",
              im_diff * 255)
  cv2.imwrite("outputs/composit.png", new_fore * 255)
  print('Press q to close all windows!')

  while True:

    """
    Show images
    """
    cv2.imshow("Calculated alpha map", new_alpha_map)
    cv2.imshow("The difference between the calculated alpha map "
               "and the ground truth", diff)
    cv2.imshow("Estimated image", final_im)
    cv2.imshow("The difference between estimated image and origninal image",
               im_diff)
    cv2.imshow("Composit foreground image", new_fore)

    if cv2.waitKey(0) & 0xFF == ord('q'):
      cv2.destroyAllWindows()
      break