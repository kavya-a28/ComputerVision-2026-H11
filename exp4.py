import numpy as np
import cv2
import matplotlib.pyplot as plt
image = cv2.imread("dog.jpg", cv2.IMREAD_GRAYSCALE)

def add_gaussian_noise(input_image, mean=0, sigma=25):
    noise = np.zeros(input_image.shape, dtype=np.float32)
    cv2.randn(noise, mean, sigma)
    noisy_image = input_image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

noisy_image = add_gaussian_noise(image,mean=0,sigma=75)

plt.figure(figsize=(6, 6))
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(noisy_image, cmap="gray")
plt.title("Image with Gaussian Noise")
plt.axis("off")
plt.show()

opencv_result = cv2.GaussianBlur(noisy_image,(5, 5),0)
plt.figure(figsize=(6, 6))
plt.imshow(opencv_result, cmap="gray")
plt.title("Gaussian Filtered Image - OpenCV")
plt.axis("off")
plt.show()

def create_gaussian_kernel(kernel_size=5, sigma=1.0):
    padding = kernel_size // 2
    x = np.arange(-padding, padding + 1)
    y = np.arange(-padding, padding + 1)
    x, y = np.meshgrid(x, y)
    # Gaussian formula
    kernel = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) /(2 * sigma ** 2))

    # Normalize kernel so that sum = 1
    kernel = kernel / np.sum(kernel)
    return kernel

gaussian_kernel = create_gaussian_kernel(kernel_size=5,sigma=1.0)

print("Gaussian Kernel:")
print(gaussian_kernel)
print("\nSum of Kernel:")
print(np.sum(gaussian_kernel))

def apply_gaussian_filter(input_image, kernel):
    padding = kernel.shape[0] // 2
    padded_image = np.pad(input_image,pad_width=padding,mode="reflect")
    height, width = input_image.shape
    filtered_image = np.zeros((height, width),dtype=np.float32)

    for i in range(height):
        for j in range(width):
            # Extract window
            window = padded_image[i:i + kernel.shape[0],j:j + kernel.shape[1]]

            # Multiply window with Gaussian kernel and calculate weighted sum
            result = np.sum(window * kernel)
            filtered_image[i, j] = result

    filtered_image = np.clip(filtered_image,0,255)
    return filtered_image.astype(np.uint8)

manual_result = apply_gaussian_filter(noisy_image,gaussian_kernel)

plt.figure(figsize=(6, 6))
plt.imshow(manual_result, cmap="gray")
plt.title("Gaussian Filtered Image - Manual")
plt.axis("off")
plt.show()