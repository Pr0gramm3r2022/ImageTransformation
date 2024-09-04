import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sklearn as sk
import torch
import cv2

print(cv2.__version__)

banker = cv2.imread("banker.jpeg")
bankerArray = np.asarray(banker)
#make into a numpy array
fourier = cv2.imread("fourierspectrum.pgm")
cv2.imshow("banker", banker)
banker_image = Image.open("banker.jpeg")
fourier_image = Image.open("fourierspectrum.pgm")

plt.imshow(banker_image, cmap = 'gray', vmin = 0, vmax = 255)
#plt.imshow(fourier_image, cmap = 'gray', vmin = 0, vmax = 255)
plt.show(banker_image)
#plt.imshow(banker_image)


