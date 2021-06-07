import numpy as np
import skimage as ski
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("images//all_souls_000026.jpg",0) 
img = ski.util.random_noise(img, mode="s&p")

dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log((cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1 ]))+1) 

rows, cols = img.shape
crow, ccol = int(rows/2), int(cols/2)

mask = np.ones((rows,cols,2), np.uint8)
r = 80
center =[crow,ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0])**2 + (y - center[1]) **2 <= r*r #eq da circunferencia
mask[mask_area] = 0
# cv2.imshow("Mascara",mask*255)

fshift = dft_shift*mask
fshift_mask_mag = 20*np.log(cv2.magnitude(fshift[:,:,0], fshift[:,:,1]))
f_ishift = np.fft.ifftshift(fshift)

imgfiltrada = cv2.idft(f_ishift)
imgfiltrada = cv2.magnitude(imgfiltrada[:,:,0], imgfiltrada[:,:,1])
# cv2.imshow("Teste", imgfiltrada)
# print(img.shape)
# f = np.fft.fft2(img)
# fshift = np.fft.fftshift(f)
# # fcomplex = fshift[:,:,0]*1j + f_shifted[:,:,1]
# magnitude_spectrum = 20*np.log(np.abs(fshift))

# cv2.imshow("Espectro", magnitude_spectrum)
fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(img, cmap='gray')
ax1.title.set_text("Entrada")
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(magnitude_spectrum, cmap='gray')
ax2.title.set_text("Espectro de Fourier")
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(fshift_mask_mag, cmap='gray')
ax3.title.set_text("Espectro com mascara")
ax4 = fig.add_subplot(2,2,4)
ax4.imshow(imgfiltrada, cmap='gray')
ax4.title.set_text("Resultado")
plt.show()
cv2.waitKey()


#-------------------------------

# laplacian=np.array([[0, 1, 0],
#                     [1,-4, 1],
#                     [0, 1, 0]])
# fft_laplacian = np.fft.fft2(laplacian)
# laplacianshift =  np.fft.fftshift(fft_laplacian)
# laplacian_spectrum = 20*np.log(np.abs(laplacianshift))

# filtered = laplacianshift*fcomplex
# inv_img = np.fft.ifft2(filtered)
# filtered_img = np.abs(inv_img)
# filtered_img -= filtered_img.min()
# filtered_img = filtered_img*255 / filtered_img.max()
# filtered_img = filtered_img.astype(np.uint8)
# cv2.imshow("AA", filtered_img)