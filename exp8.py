import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read grayscale image
img = cv2.imread("building.jpg", cv2.IMREAD_GRAYSCALE)

# Sobel kernels
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

sobel_y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=np.float32)

# Manual convolution
def convolution(image, kernel):
    H, W = image.shape
    K = kernel.shape[0]
    p = K // 2
    padded = np.pad(image, p, mode="reflect")
    output = np.zeros((H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            window = padded[i:i+K, j:j+K]
            output[i, j] = np.sum(window * kernel)
    return output

# Sobel edge detection
gx = convolution(img, sobel_x)
gy = convolution(img, sobel_y)
magnitude = np.sqrt(gx**2 + gy**2)
magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

# Otsu thresholding
_, binary = cv2.threshold(
    magnitude, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# (b1) Cartesian Hough transform
# Line equation: y = mx + c
def hough_cartesian(edge_image):
    H, W = edge_image.shape
    # Quantized slope and intercept values
    slopes = np.arange(-2.0, 2.01, 0.1)
    intercepts = np.arange(-H, H + 1, 2)
    accumulator = np.zeros(
        (len(slopes), len(intercepts)), dtype=np.uint32
    )
    ys, xs = np.where(edge_image > 0)
    # Each edge point votes for possible (m, c) pairs
    for x, y in zip(xs, ys):
        for mi, m in enumerate(slopes):
            c = y - m * x
            ci = int(np.argmin(np.abs(intercepts - c)))
            accumulator[mi, ci] += 1
            
    # Find accumulator peaks
    result = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)
    peaks = []
    for _ in range(5):
        mi, ci = np.unravel_index(
            np.argmax(accumulator), accumulator.shape
        )
        votes = accumulator[mi, ci]
        if votes < 100:
            break
        m = slopes[mi]
        c = intercepts[ci]
        peaks.append((m, c, int(votes)))
        
        # Suppress nearby accumulator cells
        accumulator[
            max(0, mi-1):min(len(slopes), mi+2),
            max(0, ci-2):min(len(intercepts), ci+3)
        ] = 0
        
    # Draw the detected Cartesian lines
    for m, c, votes in peaks:
        points = []
        for x in [0, W-1]:
            y = m * x + c
            if 0 <= y < H:
                points.append((int(x), int(y)))
        for y in [0, H-1]:
            if abs(m) > 1e-9:
                x = (y - c) / m
                if 0 <= x < W:
                    points.append((int(x), int(y)))
        if len(points) >= 2:
            cv2.line(
                result, points[0], points[1],
                (0, 255, 0), 2
            )
    return result, peaks

# (b2) Polar Hough transform
# Line equation: rho = x*cos(theta) + y*sin(theta)
def hough_polar(edge_image):
    H, W = edge_image.shape
    thetas = np.deg2rad(np.arange(0, 180, 1))
    max_rho = int(np.ceil(np.hypot(H-1, W-1)))
    rhos = np.arange(-max_rho, max_rho + 1, 1)
    accumulator = np.zeros(
        (len(rhos), len(thetas)), dtype=np.uint32
    )
    ys, xs = np.where(edge_image > 0)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    
    # Each edge point votes for possible (rho, theta) pairs
    for x, y in zip(xs, ys):
        rho_values = x * cos_t + y * sin_t
        rho_indices = np.rint(rho_values).astype(int) + max_rho
        for ti, ri in enumerate(rho_indices):
            accumulator[ri, ti] += 1
            
    # Find accumulator peaks
    result = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)
    peaks = []
    acc_copy = accumulator.copy()
    for _ in range(10):
        ri, ti = np.unravel_index(
            np.argmax(acc_copy), acc_copy.shape
        )
        votes = acc_copy[ri, ti]
        if votes < 30:
            break
        rho = rhos[ri]
        theta = thetas[ti]
        peaks.append((rho, theta, int(votes)))
        
        # Suppress nearby accumulator cells
        r0, r1 = max(0, ri-3), min(len(rhos), ri+4)
        t_indices = [
            (ti + dt) % len(thetas) for dt in range(-3, 4)
        ]
        acc_copy[r0:r1, t_indices] = 0
        
    # Draw the detected polar lines
    for rho, theta, votes in peaks:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (
            int(x0 + 2000 * (-b)),
            int(y0 + 2000 * a)
        )
        pt2 = (
            int(x0 - 2000 * (-b)),
            int(y0 - 2000 * a)
        )
        cv2.line(result, pt1, pt2, (0, 0, 255), 2)
    return result, peaks

# Run both methods
cartesian, cartesian_peaks = hough_cartesian(binary)
polar, polar_peaks = hough_polar(binary)

print("Cartesian lines (slope, intercept, votes):")
for line in cartesian_peaks:
    print(line)

print("\nPolar lines (rho, theta in radians, votes):")
for line in polar_peaks:
    print(line)

# Display results
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Grayscale Image", pad=12)
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(binary, cmap="gray")
plt.title("Sobel Edge Image", pad=12)
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(cartesian, cv2.COLOR_BGR2RGB))
plt.title("Manual Cartesian Hough Transform", pad=12)
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(polar, cv2.COLOR_BGR2RGB))
plt.title("Manual Polar Hough Transform", pad=12)
plt.axis("off")

plt.tight_layout(pad=3.0, h_pad=4.0)
plt.show()
