from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image = Image.open("cameraman.tif").convert("L")
image = np.array(image)
histogram = np.zeros(256, dtype=int)
for pixel in image.flatten():
    histogram[pixel] += 1

plt.figure(figsize=(10, 5))
plt.bar(range(256), histogram, width=1)
plt.xlabel("Gray Levels")
plt.ylabel("Number of Pixels")
plt.title("Histogram of Cameraman Image")
plt.xlim(0, 255)
plt.grid()
plt.show()

manual_threshold = 95
manual_segmented = np.zeros_like(image)

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if image[i, j] > manual_threshold:
            manual_segmented[i, j] = 255
        else:
            manual_segmented[i, j] = 0

#OTSU METHOD
total_pixels = image.size
total_sum = 0
for gray_level in range(256):
    total_sum += gray_level * histogram[gray_level]

background_weight = 0
background_sum = 0
maximum_variance = 0
otsu_threshold = 0

for threshold in range(256):
    background_weight += histogram[threshold]
    if background_weight == 0:
        continue
    foreground_weight = total_pixels - background_weight
    if foreground_weight == 0:
        break
    background_sum += threshold * histogram[threshold]
    background_mean = background_sum / background_weight
    foreground_sum = total_sum - background_sum
    foreground_mean = foreground_sum / foreground_weight
    between_class_variance = (background_weight *foreground_weight 
                              *(background_mean - foreground_mean) ** 2)

    # Find maximum variance
    if between_class_variance > maximum_variance:
        maximum_variance = between_class_variance
        otsu_threshold = threshold

print("Manual Threshold =", manual_threshold)
print("Otsu Threshold =", otsu_threshold)

otsu_segmented = np.zeros_like(image)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if image[i, j] > otsu_threshold:
            otsu_segmented[i, j] = 255
        else:
            otsu_segmented[i, j] = 0

plt.figure(figsize=(10, 5))

plt.bar(range(256), histogram, width=1)

plt.axvline(manual_threshold,linestyle="--",label="Manual Threshold = " + str(manual_threshold))
plt.axvline(otsu_threshold,linestyle="--",label="Otsu Threshold = " + str(otsu_threshold))

plt.xlabel("Gray Levels")
plt.ylabel("Number of Pixels")
plt.title("Histogram with Manual and Otsu Thresholds")

plt.xlim(0, 255)
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
# Original image
plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Cameraman")
plt.axis("off")

# Manual threshold
plt.subplot(1, 3, 2)
plt.imshow(manual_segmented, cmap="gray")
plt.title("Manual Threshold = " + str(manual_threshold))
plt.axis("off")

# Otsu threshold
plt.subplot(1, 3, 3)
plt.imshow(otsu_segmented, cmap="gray")
plt.title("Otsu Threshold = " + str(otsu_threshold))
plt.axis("off")

plt.tight_layout()
plt.show()
