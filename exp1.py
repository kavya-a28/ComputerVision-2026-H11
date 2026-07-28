import numpy as np
import cv2
import matplotlib.pyplot as plt
print("Reading the image")
img = cv2.imread("dog.jpg")
print("Original Image Shape (Height, Width, Channels):", img.shape)
cv2.imshow("Original Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

resize=cv2.resize(img,(100,100))
print("Resized Image Shape:", resize.shape)
cv2.imshow("Resized Image",resize)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Datatype of image:", img.dtype)

print("Cropping the image")
crop=img[90:180,50:200]
print("Cropped Image Shape:", crop.shape)
cv2.imshow("Cropped image",crop)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Reading the pixel value at (150,150)")
b, g, r = img[150,150]
print("Blue :", b)
print("Green:", g)
print("Red  :", r)

print("Changing pixel (150,150) to Black")
img[150,150]=[0,0,0]
print("New Pixel Value:", img[150,150])
cv2.imshow("Black pixel image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Converting BGR to RGB because Matplotlib uses RGB.")
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title("Image in Matplotlib (RGB)")
plt.show()

grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print("Gray Image Shape:", grey.shape)
cv2.imshow("Gray image in OpenCV",grey)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Displaying grayscale image using Matplotlib")
plt.imshow(grey, cmap="gray")
plt.title("Gray Image in Matplotlib")
plt.show()

print("Other functions")
print("Rotate Image")
rotate = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
print("Original Shape :", img.shape)
print("Rotated Shape  :", rotate.shape)
cv2.imshow("Rotated Image", rotate)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Flip Image")
flip = cv2.flip(img, 1)
print("Image flipped horizontally.")
cv2.imshow("Flipped Image", flip)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Blur Image")
blur = cv2.GaussianBlur(img, (11,11), 0)
print("Applied Gaussian Blur")
cv2.imshow("Blurred Image", blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Save Image")
cv2.imwrite("new_dog.jpg", img)
print("Image saved successfully!")

