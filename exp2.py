import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("dog.jpg",cv2.IMREAD_GRAYSCALE)
print(image.shape)
plt.imshow(image,cmap="gray")
plt.title("Gray Image")
plt.show()

def add_salt_pepper_noise(image,probability=0.3):
    noisy = image.copy()
    random = np.random.rand(*image.shape)
    noisy[random<probability/2]=0
    noisy[random>1-probability/2]=255
    return noisy

prob =0.2
noisy = add_salt_pepper_noise(image,prob)
plt.imshow(noisy,cmap="gray")
plt.title("Added salt and pepper noise")
plt.show()


m, n = image.shape
output = image.copy()
for i in range(1, m-1):
    for j in range(1, n-1):
        window = image[i-1:i+2, j-1:j+2]
        output[i, j] = np.mean(window) 

plt.imshow(output, cmap="gray")
plt.title("Removed Salt and Pepper noise")
plt.show()

m, n = image.shape
display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for i in range(1, m-1):
    for j in range(1, n-1):
        temp = display.copy()
        cv2.rectangle(temp,
            (j-1, i-1),      # Top-left corner
            (j+1, i+1),      # Bottom-right corner
            (0, 255, 0),     # Green color
            5                # Thickness
        )
        cv2.imshow("Moving 3x3 Window", temp)
        key = cv2.waitKey(30)   
        if key == 27:          
            break
cv2.destroyAllWindows()