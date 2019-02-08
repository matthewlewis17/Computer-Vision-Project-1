import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from scipy import ndimage, misc

# Read image of puppy and resize by 1/4

path = '\\Users\\legit\\Documents\\391 lab 1\\'
img = cv2.imread(path + 'dog noisy.JPG')
small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

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
#colData = smallNoisy[0:smallNoisy.shape[0], col, 0]
colData = small[0:smallNoisy.shape[0], col, 0]

# Plot the column data as a function
xvalues = np.linspace(0, len(colData) - 1, len(colData))
plt.plot(xvalues, colData, 'b')
#plt.savefig('/Users/paucavp1/Temp/function.png', bbox_inches='tight')
#plt.clf()
plt.show()

# Compute the 1-D Fourier transform of colData
F_colData = np.fft.fft(colData.astype(float))

# Plot the magnitude of the Fourier coefficients as a stem plot
# Notice the use off fftshift() to center the low frequency coefficients around 0
xvalues = np.linspace(0, len(colData), len(colData))
markerline, stemlines, baseline = plt.stem(xvalues, (np.abs(F_colData)), 'g')
plt.setp(markerline, 'markerfacecolor', 'g')
plt.setp(baseline, 'color','r', 'linewidth', 0.5)
plt.show()

print(np.abs(F_colData))

# Convert small image to grayscale so that a 3D plot is easier to make
graySmall = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
cv2.imwrite("grayImage.jpg", graySmall)
# create the x and y coordinate arrays (here we just use pixel indices)
xx, yy = np.mgrid[0:graySmall.shape[0], 0:graySmall.shape[1]]

# Take the 2-D DFT and plot the magnitude of the corresponding Fourier coefficients
F2_graySmall = np.fft.fft2(graySmall.astype(float))
Y = (np.linspace(-int(graySmall.shape[0]/2), int(graySmall.shape[0]/2)-1, graySmall.shape[0]))
X = (np.linspace(-int(graySmall.shape[1]/2), int(graySmall.shape[1]/2)-1, graySmall.shape[1]))
X, Y = np.meshgrid(X, Y)

# Plot the magnitude and the log(magnitude + 1) as images (view from the top)
magnitudeImage = np.fft.fftshift(np.abs(F2_graySmall))
magnitudeImage = magnitudeImage / magnitudeImage.max()   # scale to [0, 1]
magnitudeImage = ski.img_as_ubyte(magnitudeImage)
logMagnitudeImage = np.fft.fftshift(np.log(np.abs(F2_graySmall)+1))
logMagnitudeImage = logMagnitudeImage / logMagnitudeImage.max()   # scale to [0, 1]
logMagnitudeImage = ski.img_as_ubyte(logMagnitudeImage)

# Explore the Butterworth filter
# U and V are arrays that give all integer coordinates in the 2-D plane
#  [-m/2 , m/2] x [-n/2 , n/2].
# Use U and V to create 3-D functions over (U,V)
U = (np.linspace(-int(graySmall.shape[0]/2), int(graySmall.shape[0]/2)-1, graySmall.shape[0]))
V = (np.linspace(-int(graySmall.shape[1]/2), int(graySmall.shape[1]/2)-1, graySmall.shape[1]))
U, V = np.meshgrid(U, V)
# The function over (U,V) is distance between each point (u,v) to (0,0)
D = np.sqrt(X*X + Y*Y)
# create x-points for plotting
xval = np.linspace(-int(graySmall.shape[1]/2), int(graySmall.shape[1]/2)-1, graySmall.shape[1])
# Specify a frequency cutoff value as a function of D.max()
D0 = 0.25 * D.max()

# The ideal lowpass filter makes all D(u,v) where D(u,v) <= 0 equal to 1
# and all D(u,v) where D(u,v) > 0 equal to 0
idealLowPass = D <= D0

# Filter our small grayscale image with the ideal lowpass filter
# 1. DFT of image
print(graySmall.dtype)
FTgraySmall = np.fft.fft2(graySmall.astype(float))
# 2. Butterworth filter is already defined in Fourier space
# 3. Elementwise product in Fourier space (notice fftshift of the filter)
FTgraySmallFiltered = FTgraySmall * np.fft.fftshift(idealLowPass)
# 4. Inverse DFT to take filtered image back to the spatial domain
graySmallFiltered = np.abs(np.fft.ifft2(FTgraySmallFiltered))

# Save the filter and the filtered image (after scaling)
idealLowPass = ski.img_as_ubyte(idealLowPass / idealLowPass.max())
graySmallFiltered = ski.img_as_ubyte(graySmallFiltered / graySmallFiltered.max())
cv2.imwrite("idealLowPass.jpg", idealLowPass)
cv2.imwrite("grayImageIdealLowpassFiltered.jpg", graySmallFiltered)

# Plot the ideal filter and then create and plot Butterworth filters of order
n = 1, 2, 3, 4
plt.plot(xval, idealLowPass[int(idealLowPass.shape[0]/2), :], 'c--', label='ideal')
colors='brgkmc'
for n in range(1, 5):
    # Create Butterworth filter of order n
    H = 1.0 / (1 + (np.sqrt(2) - 1)*np.power(D/D0, 2*n))
    # Apply the filter to the grayscaled image
    FTgraySmallFiltered = FTgraySmall * np.fft.fftshift(H)
    graySmallFiltered = np.abs(np.fft.ifft2(FTgraySmallFiltered))
    graySmallFiltered = ski.img_as_ubyte(graySmallFiltered / graySmallFiltered.max())
    cv2.imwrite("grayImageButterworth-n" + str(n) + ".jpg", graySmallFiltered)
    cv2.imshow('Gray Image Butterworth ' + str(n), graySmallFiltered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    H = ski.img_as_ubyte(H / H.max())
    cv2.imwrite("butter-n" + str(n) + ".jpg", H)
    cv2.imshow('No image filter n ' + str(n), H)
    cv2.waitKey(0)
    # Get a slice through the center of the filter to plot in 2-D
    slice = H[int(H.shape[0]/2), :]
    plt.plot(xval, slice, colors[n-1], label='n='+str(n))
    plt.legend(loc='upper left')

plt.show()
plt.savefig('butterworthFilters.jpg', bbox_inches='tight')