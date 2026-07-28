import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def binarize_image(image_array, threshold=128):
    return (image_array < threshold).astype(float)

def spatial_moment(binary_img, p, q):
    y_indices, x_indices = np.indices(binary_img.shape)
    return np.sum((x_indices ** p) * (y_indices ** q) * binary_img)

def central_moment(binary_img, p, q, x_c, y_c):
    y_indices, x_indices = np.indices(binary_img.shape)
    x_shifted = x_indices - x_c
    y_shifted = y_indices - y_c
    return np.sum((x_shifted ** p) * (y_shifted ** q) * binary_img)

def normalized_central_moment(mu_pq, m00, p, q):
    gamma = ((p + q) / 2.0) + 1.0
    return mu_pq / (m00 ** gamma)


def calculate_hu_moments(image_array):
    binary_img = binarize_image(image_array)

    m00 = spatial_moment(binary_img, 0, 0)
    if m00 == 0:
        return 0, 0

    m10 = spatial_moment(binary_img, 1, 0)
    m01 = spatial_moment(binary_img, 0, 1)

    x_c = m10 / m00
    y_c = m01 / m00

    mu20 = central_moment(binary_img, 2, 0, x_c, y_c)
    mu02 = central_moment(binary_img, 0, 2, x_c, y_c)
    mu21 = central_moment(binary_img, 2, 1, x_c, y_c)
    mu12 = central_moment(binary_img, 1, 2, x_c, y_c)

    mu30 = central_moment(binary_img, 3, 0, x_c, y_c)
    mu03 = central_moment(binary_img, 0, 3, x_c, y_c)

    eta20 = normalized_central_moment(mu20, m00, 2, 0)
    eta02 = normalized_central_moment(mu02, m00, 0, 2)
    eta21 = normalized_central_moment(mu21, m00, 2, 1)
    eta12 = normalized_central_moment(mu12, m00, 1, 2)

    eta30 = normalized_central_moment(mu30, m00, 3, 0)
    eta03 = normalized_central_moment(mu03, m00, 0, 3)

    phi1 = eta20 + eta02
    phi2 = (eta30 + eta12)**2 + (eta21 + eta03)**2

    return phi1, phi2


def classify_image(test_phi, reference_features):
    min_dist = float('inf')
    best_match = None

    phi1_test, phi2_test = test_phi

    for ref_name, ref_phi in reference_features.items():
        phi1_ref, phi2_ref = ref_phi

        distance = np.sqrt((phi1_test - phi1_ref)**2 + (phi2_test - phi2_ref)**2)

        if distance < min_dist:
            min_dist = distance
            best_match = ref_name

    return best_match, min_dist

def main():
    reference_files = ['a.bmp', 'b.bmp', 'f.bmp', 'm.bmp']
    test_files = ['a1.bmp', 'a2.bmp', 'b1.bmp', 'f1.bmp', 'm1.bmp', 'm2.bmp']

    reference_features = {}
    loaded_ref_images = {}
    loaded_test_images = {}
    classification_results = {}

    print("--- 1. CALCULATION OF REFERENCE IMAGES ---")
    for file in reference_files:
        img = Image.open(file).convert('L')
        img_array = np.array(img)
        loaded_ref_images[file] = img_array

        phi1, phi2 = calculate_hu_moments(img_array)
        reference_features[file] = (phi1, phi2)
        print(f"Reference '{file}': Ф1 = {phi1:.8f}, Ф2 = {phi2:.8f}")

    print("\n--- 2. CLASSIFICATION OF TEST IMAGES ---")
    print(f"{'File':<10} | {'Ф1':<12} | {'Ф2':<12} | {'Result (Reference)':<20} | {'Distance'}")
    print("-" * 75)

    for file in test_files:
        img = Image.open(file).convert('L')
        img_array = np.array(img)
        loaded_test_images[file] = img_array

        test_phi = calculate_hu_moments(img_array)

        best_match, distance = classify_image(test_phi, reference_features)
        classification_results[file] = best_match

        print(f"{file:<10} | {test_phi[0]:.8f} | {test_phi[1]:.8f} | {best_match:<20} | {distance:.8f}")

    if loaded_ref_images and loaded_test_images:
        num_refs = len(loaded_ref_images)
        num_tests = len(loaded_test_images)

        cols = max(num_refs, num_tests)
        fig, axes = plt.subplots(2, cols, figsize=(15, 8))
        fig.suptitle('Results of classification by moments', fontsize=16)

        for ax in axes.flat:
            ax.axis('off')

        for i, (name, img_arr) in enumerate(loaded_ref_images.items()):
            axes[0, i].imshow(img_arr, cmap='gray')
            axes[0, i].set_title(f"Reference: {name}", color='blue')
            axes[0, i].axis('on')
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])

        for i, (name, img_arr) in enumerate(loaded_test_images.items()):
            match_name = classification_results.get(name, "Error")
            axes[1, i].imshow(img_arr, cmap='gray')
            axes[1, i].set_title(f"Test: {name}\nClass: {match_name}", color='green')
            axes[1, i].axis('on')
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])

        plt.tight_layout()
        plt.show()
main()
