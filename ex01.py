import numpy as np
import skimage as ski
import cv2


def Media(img, n):
    kernel = np.ones((n,n), np.float32)/(n**2)
    return cv2.filter2D(img, -1, kernel)
    
def Mediana(img, n):
    return cv2.medianBlur(img, n)

def PassaAlta(img):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)

def Gaussiano(img, n):
    # x = cv2.getGaussianKernel(5, 10)
    # print(x)
    return cv2.GaussianBlur(img, (n,n), 0)

img = cv2.imread("images//all_souls_000026.jpg",0) 
noise = ski.util.random_noise(img, mode="pepper")
cv2.imshow("Ruido", noise)
imgMedia = Media(img, 3)
imgMedian = Mediana(img, 3)
imgAlta = PassaAlta(img)
imgGaussian = Gaussiano(img, 5)

cv2.imshow("Resultado - Media", imgMedia)
cv2.imshow("Resultado - Mediana", imgMedian)
cv2.imshow("Resultado - Passa Alta", imgAlta)
cv2.imshow("Resultado - Guassiano", imgGaussian)

cv2.imwrite("results//ex01Media.jpg", imgMedia)
cv2.imwrite("results//ex01Median.jpg", imgMedian)
cv2.imwrite("results//ex01Alta.jpg", imgAlta)
cv2.imwrite("results//ex01Guassian.jpg", imgGaussian)
cv2.waitKey()
