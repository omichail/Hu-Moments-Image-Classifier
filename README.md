# Hu-Moments-Image-Classifier

## Description
The primary goal of this project is to implement a mathematically rigorous shape recognition and image classification system from scratch using Hu Invariant Moments. Driven by a deep interest in foundational Computer Vision (CV), this project demonstrates how to identify and match objects regardless of their translation, scale, or rotation without relying on heavy, black-box Machine Learning models. It addresses the need for computationally lightweight and fully transparent pattern recognition algorithms.

### Algorithm Architecture & Mechanics
The classification pipeline processes binary images through a sequence of strict mathematical transformations:
* **Image Binarization:** Converts raw image data into a binary matrix to isolate the shape's footprint.
* **Moment Computations:** Calculates spatial moments to find the shape's centroid, followed by central and normalized central moments to achieve translation and scale invariance.
* **Hu Invariants Extraction:** Computes the first two Hu Moments ($\phi_1, \phi_2$) which serve as a unique, rotation-invariant mathematical signature for each shape.
* **Distance-Based Classification:** Compares the extracted signatures of test images against reference images using Euclidean distance to find the closest match.

### Technologies Used
* **Python** — core programming language.
* **NumPy** — heavily utilized for efficient 2D array manipulation, coordinate indexing (`np.indices`), and vectorized mathematical summation for moment extraction.
* **Pillow (PIL)** — used for image I/O operations and basic thresholding.
* **Matplotlib** — applied for generating a structured visual dashboard that displays the reference images alongside the test images and their classification predictions.

### Results
The custom algorithm successfully calculates the invariant functions for both reference and test datasets. It precisely classifies various distorted or modified test characters (e.g., `a1.bmp`, `m2.bmp`) by matching their geometric properties to the original references. Unlike data-hungry Neural Networks, this deterministic approach achieves accurate classification using only a single reference image per class, showcasing extreme computational efficiency and mathematical stability.

### Visualization
The script automatically generates a visual grid comparing the reference shapes with the classified test inputs, highlighting the predicted matches.

<p align="center">
  <b>Hu Moments Classification Results</b><br><br>
  <img src="assets/classification_results.png" width="80%" alt="Image Classification Results"><br><br>
  <sub>Reference images vs. Test images with computed algorithmic matches</sub>
</p>

## Quick Start Guide

### 1. Download the Files
Save the Python script `Lab3.py` and your reference/test image datasets (`.bmp` files) to your local machine.

### 2. Install Dependencies
Ensure you have Python 3.8+ installed. Open your terminal or command prompt and execute:
```bash
pip install numpy matplotlib pillow
```
### 3. Execute the code
Run the main script to start the calculations and render the visualization:
```bash
python Image-Classifier.py
```
