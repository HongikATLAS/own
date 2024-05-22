import cv2
import numpy as np
from matplotlib import pyplot as plt
# Convert to depth image

imgL = cv2.imread('C:/Users/kchw3_r2l9a77/OneDrive/Desktop/left.jpg',0)
imgR = cv2.imread('C:/Users/kchw3_r2l9a77/OneDrive/Desktop/right.jpg',0)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

disparity = stereo.compute(imgL,imgR)

plt.imshow(disparity,'gray')

plt.show()

