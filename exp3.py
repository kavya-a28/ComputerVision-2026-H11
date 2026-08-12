import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("dog.jpg", cv2.IMREAD_GRAYSCALE)
print("Original Image Shape:", image.shape)
plt.imshow(image, cmap="gray")
plt.title("Original Grayscale Image")
plt.axis("off")
plt.show()
def add_noise(input_image, noise_probability=0.3):
    noisy_image = input_image.copy()
    random_values = np.random.rand(*input_image.shape)
    noisy_image[random_values < noise_probability / 2] = 0
    noisy_image[random_values > 1 - noise_probability / 2] = 255
    return noisy_image

noise_probability = 0.3
noisy_image = add_noise(image,noise_probability)
plt.imshow(noisy_image, cmap="gray")
plt.title("Salt and Pepper Noisy Image")
plt.axis("off")
plt.show()

def apply_median_filter(input_image, kernel_size=4, stride=2):
    padding = kernel_size // 2
    padded_image = np.pad(input_image,pad_width=padding,mode="reflect")
    image_height, image_width = input_image.shape
    output_height = ((image_height + 2 * padding - kernel_size)// stride) + 1
    output_width = ((image_width + 2 * padding - kernel_size)// stride) + 1
    filtered_image = np.zeros((output_height, output_width),dtype=np.uint8)
    for row_index in range(output_height):
        for column_index in range(output_width):
            start_row = row_index * stride
            start_column = column_index * stride
            pixel_window = padded_image[
                start_row:start_row + kernel_size,
                start_column:start_column + kernel_size
            ]
            filtered_image[row_index, column_index] = np.median(pixel_window)
    return filtered_image

median_result = apply_median_filter(noisy_image,kernel_size=3,stride=1)
plt.imshow(median_result, cmap="gray")
plt.title("Median Filter")
plt.axis("off")
plt.show()

def apply_mean_filter(input_image, kernel_size=4, stride=2):
    padding = kernel_size // 2
    padded_image = np.pad(input_image,pad_width=padding,mode="reflect")
    image_height, image_width = input_image.shape
    output_height = ((image_height + 2 * padding - kernel_size)// stride) + 1
    output_width = ((image_width + 2 * padding - kernel_size)// stride) + 1
    filtered_image = np.zeros((output_height, output_width),dtype=np.uint8)
    for row_index in range(output_height):
        for column_index in range(output_width):
            start_row = row_index * stride
            start_column = column_index * stride
            pixel_window = padded_image[
                start_row:start_row + kernel_size,
                start_column:start_column + kernel_size
            ]
            filtered_image[row_index, column_index] = np.mean(pixel_window)
    return filtered_image

mean_result = apply_mean_filter(noisy_image,kernel_size=3,stride=1)
plt.imshow(mean_result, cmap="gray")
plt.title("Mean Filter")
plt.axis("off")
plt.show()