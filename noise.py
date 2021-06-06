import numpy as np
import skimage as ski
import cv2


img = cv2.imread("images//all_souls_000026.jpg",0) 
print(img.shape)
noise = ski.util.random_noise(img, mode="pepper")
# cv2.imwrite("sp_noise.jpg", noise)
cv2.imshow("Ruido", noise)
cv2.waitKey()