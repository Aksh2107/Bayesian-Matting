#BAYESIAN MATTING 

This project implements the Bayesian Matting technique described in Yung-Yu Chuang, Brian Curless, David H. Salesin and Richard Szeliski. A Bayesian Approach to Digital Matting. In Proceedings of IEEE Computer Vision and Pattern Recognition (CVPR 2001), Vol. II, 264-271, December 2001 [1].
Our implementation is slight different from Paper :
[1] Yung-Yu Chuang, B. Curless, D. H. Salesin and R. Szeliski, "A Bayesian approach to digital matting," Proceedings of the 2001 IEEE Computer Society Conference on Computer Vision and Pattern Recognition. CVPR 2001, 2001, pp. II-II
[2] http://www.alphamatting.com/datasets.php

We test the performance of our algorithm using three performance metrics viz. MSE (between original image and the estimated image & alpha matte and ground truth), PSNR (between original image and the estimated image), and SSIM (between original image and the estimated image & alpha matte and ground truth).

1. We didn't implemet the clustering methodology 
2. We experimented with size of the kernel, thus taking into account number of pixels
3. Assigning weights to the kernel depending upon the size of the kernel greater than 50 x 50

Running the DEMO : 
- Download 'Byaesian Matting' folder
- Run pip install -r requirements.txt in terminal
- Add the input, ground-truth and tri-map images into the input folder
- Run the 'Main.py' in the terminal
- Insert the name of the images according to the input prompt (don't need to add path, it will read the input folder's path)
- The final result will be generation of five output images that will be stored in the output folder
- The performance metrics for the input images will be displayed

Function of each '.py' file :

1] Main.py 
=> calls all the function implemented and run

2] getKernel.py
=> this file constructs the kernel depending upon the set number of pixels required within the kernel.

3] kernelg.py
=> this file constructs the weight of the kernel

4] getFbarBbar.py
=> this file computes the mean 'foreground' value and mean 'background' value

5] getFcovBcov.py
=> this file computes the covariance of the 'foreground' and covariance of 'background'

6] matrixequation.py
=> this file computes the mathematical equations necesssary for the algorithm to run 

7] BayesianMatting.py
=> this file returns the alpha matte, difference between the ground truth and the alpha matte, the estimated image, difference between estimated and    original image and lastly the composited image

8] ShowNwriteIm.py
=> this file displays and writes the results








