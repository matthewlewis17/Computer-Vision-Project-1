import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski

# Read image of Ima and resize by 1/4
path = '\\Users\\legit\\Documents\\391 lab 1\\'
img = cv2.imread(path + 'dog noisy.JPG')
k = int(input('Enter filter size k: '))
small = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
originalImg = small

# Create noisy image as g = f + sigma * noise
# with noise scaled by sigma = .2 max(f)/max(noise)
noise = np.random.randn(small.shape[0], small.shape[1])
smallNoisy = np.zeros(small.shape, np.float64)
sigma = 0.2 * small.max()/noise.max()
# Color images need noise added to all channels
if len(small.shape) == 2:
    smallNoisy = small + sigma * noise
else:
    smallNoisy[:, :, 0] = small[:, :, 0] + sigma * noise
    smallNoisy[:, :, 1] = small[:, :, 1] + sigma * noise
    smallNoisy[:, :, 2] = small[:, :, 2] + sigma * noise

# Calculate the index of the middle column in the image
col = int(smallNoisy.shape[1]/2)
# Obtain the image data for this column
colData = smallNoisy[0:smallNoisy.shape[0], col, 0]

# Plot the column data as a stem plot of xvalues vs colData
xvalues = np.linspace(0, len(colData) - 1, len(colData))
markerline, stemlines, baseline = plt.stem(xvalues, colData, 'b')
plt.setp(markerline, 'markerfacecolor', 'b')
plt.setp(baseline, 'color', 'r', 'linewidth', 0.5)
plt.title('Plot of original image')
plt.savefig('Original image plot.png')
plt.show()



# Create a 1-D filter of length 3: [1/3, 1/3, 1/3] and apply to column data
windowLen = k
w = np.ones(windowLen, 'd')  # vector of ones
w = w / w.sum()
y = np.convolve(w, colData, mode='valid')

# Plot the filtered column data as a stem plot
xvalues = np.linspace(0, len(y)-1, len(y))
markerline, stemlines, baseline = plt.stem(xvalues, y, 'g')
plt.setp(markerline, 'markerfacecolor', 'g')
plt.setp(baseline, 'color','r', 'linewidth', 0.5)
plt.title('Plot of denoised image')
plt.savefig('Denoised image plot.png')
plt.show()


# Create a 2-D box filter of size 3 x 3 and scale so that sum adds up to 1
w = np.ones((windowLen, windowLen), np.float32)
w = w / w.sum()
filteredimg = np.zeros(smallNoisy.shape, np.float64)  # array for filteredimg image
# Apply the filter to each channel
filteredimg[:, :, 0] = cv2.filter2D(smallNoisy[:, :, 0], -1, w)
filteredimg[:, :, 1] = cv2.filter2D(smallNoisy[:, :, 1], -1, w)
filteredimg[:, :, 2] = cv2.filter2D(smallNoisy[:, :, 2], -1, w)

filteredimg_copy = filteredimg
# 1st option: Scaled the filteredimg image back to uint8 using astype()
#filteredimg = (filteredimg + filteredimg.min()) / smallNoisy.max()  # preserve range and scale to [0, 1] if min() is negative
filteredimg = filteredimg / smallNoisy.max()  # scale to [min/max, 1]
filteredimg = filteredimg * 255 # scale to [min/max*255, 255]

cv2.imshow('filteredimg', filteredimg.astype(np.uint8))  # convert to uint8
cv2.imwrite('filteredDog.jpg', filteredimg)

cv2.imshow('originalImg', originalImg.astype(np.uint8))
cv2.imwrite('originalDog.jpg', originalImg)

gaussianImg = cv2.GaussianBlur(originalImg, (k, k), 0)
cv2.imshow('Gaussian filtered', gaussianImg.astype(np.uint8))
cv2.imwrite('Gaussian filtered.jpg', gaussianImg)

edges = cv2.Canny(originalImg,100,200)
cv2.imshow('Edge original dog', edges.astype(np.uint8))
cv2.imwrite('Edge original dog.jpg', edges)

filteredimg = np.uint8(filteredimg)
edgeSpatial = cv2.Canny(filteredimg,100,200)
cv2.imshow('Edge spatial dog', edgeSpatial.astype(np.uint8))
cv2.imwrite('Edge spatial dog.jpg', edgeSpatial)

path = '\\Users\\legit\\Documents\\391 lab 1\\'
img = cv2.imread(path + 'landscape.JPG')
landscape = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
landscape = np.uint8(landscape)
landscapeEdge = cv2.Canny(landscape,100,200)
cv2.imshow('landscape original', landscape.astype(np.uint8))
cv2.imwrite('landscape original.jpg', landscape)
cv2.imshow('landscape edge', landscapeEdge.astype(np.uint8))
cv2.imwrite('landscape edge.jpg', landscapeEdge)
cv2.waitKey(0)

#show edge detection on gaussian and normal, also add median