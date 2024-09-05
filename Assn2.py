import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional
from PIL import Image
import sklearn as sk
import torch
import cv2

#cv2 version is 4.10.0
banker = cv2.imread("banker.jpeg")
bankerArray = np.asarray(banker)
#make into a numpy array
fourier = cv2.imread("fourierspectrum.pgm")
cv2.imshow("banker", banker)

banker_image = Image.open("banker.jpeg")
bankerArray = np.asanyarray(banker_image)
print(bankerArray.flags)
print(bankerArray.dtype)
if isinstance(bankerArray, np.ndarray):
    print("banker is an ndarray")


fourier_image = Image.open("fourierspectrum.pgm")
fourierArray = np.asarray(fourier_image)
bankerTensor = torch.tensor(bankerArray)
fourierTensor = torch.tensor(fourierArray)


plt.imshow(banker_image, cmap = 'gray', vmin = 0, vmax = 255)
plt.show()
plt.imshow(fourier_image, cmap = 'gray', vmin = 0, vmax = 255)
plt.show()
#plt.imshow(fourier_image, cmap = 'gray', vmin = 0, vmax = 255)
#plt.show(banker_image)
#plt.imshow(banker_image)

# transformation function is s = T(r) = c*log(1+r)
bankerLog = torch.log(bankerTensor)
fourierLog = torch.log(fourierTensor)




def tensor_to_image(bankerLog):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)



plt.imshow(bankerLog, cmap = 'gray', vmin = 0, vmax = 255)
plt.show()

def tensor_to_image(fourierLog):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

plt.imshow(fourierLog, cmap = 'gray', vmin = 0, vmax = 255)
plt.show()

torchvision.transforms.functional.adjust_gamma(bankerTensor)
torchvision.transforms.functional.adjust_gamma(fourierTensor)


torchvision.transforms.functional.equalize(bankerTensor)
torchvision.transforms.functional.equalize(fourierTensor)