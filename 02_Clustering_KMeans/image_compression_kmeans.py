import numpy as np
import cv2 # OpenCV library
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt

def compress_image_kmeans_opencv(image_path, n_colors=64):
    try:
        # 1. Read an image and convert it into a 2D array of pixel values.
        # OpenCV reads images in BGR format by default.
        original_image_bgr = cv2.imread(image_path)

        if original_image_bgr is None:
            raise FileNotFoundError(f"Image file not found at: {image_path}")

        # Convert BGR to RGB for consistent visualization with Matplotlib
        original_image_rgb = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB)

        # We'll use the RGB version for clustering since K-Means doesn't care about order,
        # but displaying with Matplotlib expects RGB.
        image_np = original_image_rgb

        # Reshape the image into a 2D array of pixels (height * width, 3)
        height, width, _ = image_np.shape
        pixel_data = image_np.reshape(-1, 3)

        print(f"Original image shape: {image_np.shape}")
        print(f"Reshaped pixel data shape for K-Means: {pixel_data.shape}")

        # Convert pixel data to float32, as K-Means often prefers float input
        pixel_data = np.float32(pixel_data)

        # 2. Apply K-Means clustering to reduce the number of colors.
        # Using MiniBatchKMeans for efficiency.
        kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42, n_init='auto', verbose=False)
        kmeans.fit(pixel_data)

        # Get the cluster centers (the new, reduced colors)
        new_colors = kmeans.cluster_centers_

        # Assign each pixel to its nearest cluster center (new color)
        labels = kmeans.predict(pixel_data)

        # Replace each pixel with its corresponding cluster center color
        compressed_pixel_data = new_colors[labels]

        # 3. Reconstruct and visualize the compressed image.
        # Reshape the compressed pixel data back to the original image dimensions
        # Convert back to uint8 (0-255) as pixel values
        compressed_image_np_rgb = compressed_pixel_data.reshape(height, width, 3).astype(np.uint8)

        return original_image_rgb, compressed_image_np_rgb

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None

if __name__ == "__main__":
    # --- Configuration ---
    input_image_file = "/content/mountain.jpg"
    num_colors_to_reduce_to = 8  # Lower numbers mean more compression / fewer distinct colors.


    # --- Perform Compression ---
    original_img_rgb, compressed_img_rgb = compress_image_kmeans_opencv(
        input_image_file, num_colors_to_reduce_to
    )

    if original_img_rgb is not None and compressed_img_rgb is not None:
        # --- Visualization ---
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(original_img_rgb)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(compressed_img_rgb)
        plt.title(f"Compressed Image ({num_colors_to_reduce_to} colors)")
        plt.axis('off')

        plt.tight_layout()
        plt.show()


    else:
        print("Image compression failed. Please check the image path and file format.")
