import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("dog.jpg", cv2.IMREAD_GRAYSCALE)

# Sobel X kernel
sobel_x = np.array([
    [-1,  0,  1],
    [-2,  0,  2],
    [-1,  0,  1]
], dtype=np.float32)

# Sobel Y kernel
sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

def convolution(image, kernel):
    H, W = image.shape
    K = kernel.shape[0]
    padding = K // 2
    padded_image = np.pad(image,padding,mode="reflect")
    output = np.zeros((H, W),dtype=np.float32)
    for i in range(H):
        for j in range(W):
            # Extract K x K window
            window = padded_image[i:i + K,j:j + K]
            # Multiply kernel and window
            value = np.sum(window * kernel)
            output[i, j] = value
    return output

def edge_detection(image, kernel_x, kernel_y):
    gradient_x = convolution(image, kernel_x)
    gradient_y = convolution(image, kernel_y)
    square_x = gradient_x ** 2
    square_y = gradient_y ** 2
    magnitude = np.sqrt(square_x + square_y)
    magnitude = np.clip(magnitude, 0, 255)
    magnitude = magnitude.astype(np.uint8)
    return gradient_x, gradient_y, magnitude

gradient_x, gradient_y, magnitude = edge_detection(img,sobel_x,sobel_y)
gradient_x_display = np.abs(gradient_x)
gradient_y_display = np.abs(gradient_y)

gradient_x_display = np.clip(gradient_x_display,0,255).astype(np.uint8)
gradient_y_display = np.clip(gradient_y_display,0,255).astype(np.uint8)

plt.figure(figsize=(6, 5))
plt.imshow(gradient_x_display, cmap="gray")
plt.title("Sobel X (Gx)")
plt.axis("off")
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(gradient_y_display, cmap="gray")
plt.title("Sobel Y (Gy)")
plt.axis("off")
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(magnitude, cmap="gray")
plt.title("Sobel Magnitude")
plt.axis("off")
plt.show()