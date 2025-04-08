# Example code for using the color correction functions
# Copy and paste this into your Jupyter notebook

# Import necessary modules
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys

# Add src directory to path to use provided functions
sys.path.append('./src')
from cp_assgn2 import read_colorchecker_gm

# Import our color correction functions
from color_correction import (crop_color_patches, compute_average_rgb, 
                              convert_to_homogeneous, compute_affine_transform,
                              apply_affine_transform, apply_white_balance,
                              save_hdr_image, color_correct_hdr_image,
                              visualize_color_correction)

# Step 1: Load your HDR image
# Change this to your HDR image path
hdr_image_path = 'results/tiff_logarithmic_gaussian.hdr'  # Update with your best HDR image
hdr_image = cv2.imread(hdr_image_path, cv2.IMREAD_ANYDEPTH)
hdr_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Step 2: Get the ground truth color checker values
color_checker_values = read_colorchecker_gm()  # This should return a 4x6x3 matrix

# Step 3: Run the complete color correction pipeline
# Set interactive=True to select patches manually, or use saved coordinates
# For the first run, you'll need to select patches interactively
color_corrected, white_balanced, coordinates = color_correct_hdr_image(
    hdr_image, 
    color_checker_values, 
    interactive=True,  # Set to False after first run if you save coordinates
    saved_coordinates=None  # Use your saved coordinates after first run
)

# Save the coordinates for future use (optional)
np.save('color_checker_coordinates.npy', coordinates)

# Step 4: Save the color-corrected and white-balanced HDR images
save_hdr_image(color_corrected, 'color_corrected.hdr')
save_hdr_image(white_balanced, 'white_balanced.hdr')

# Step 5: Visualize the results for comparison
visualize_color_correction(hdr_image, color_corrected, white_balanced)

# ------------------------------------------------------------------------
# If you want to re-run with saved coordinates later, use this code:
# ------------------------------------------------------------------------
# # Load saved coordinates
# saved_coordinates = np.load('color_checker_coordinates.npy', allow_pickle=True)
# 
# # Run color correction with saved coordinates
# color_corrected, white_balanced, _ = color_correct_hdr_image(
#     hdr_image, 
#     color_checker_values, 
#     interactive=False,  
#     saved_coordinates=saved_coordinates
# )
# 
# # Save and visualize as before
# save_hdr_image(color_corrected, 'color_corrected.hdr')
# save_hdr_image(white_balanced, 'white_balanced.hdr')
# visualize_color_correction(hdr_image, color_corrected, white_balanced)

# ------------------------------------------------------------------------
# For more granular control, you can also use the individual functions:
# ------------------------------------------------------------------------
# # 1. Crop patches manually or using saved coordinates
# patches, coordinates = crop_color_patches(hdr_image, interactive=True)
# 
# # 2. Compute average RGB values for the patches
# avg_rgb = compute_average_rgb(patches)
# print("Average RGB values shape:", avg_rgb.shape)
# 
# # 3. Convert to homogeneous coordinates
# homogeneous_coords = convert_to_homogeneous(avg_rgb)
# print("Homogeneous coordinates shape:", homogeneous_coords.shape)
# 
# # 4. Get ground truth and reshape to 24x3
# ground_truth = color_checker_values.reshape(24, 3)
# print("Ground truth shape:", ground_truth.shape)
# 
# # 5. Compute affine transformation matrix
# transform = compute_affine_transform(homogeneous_coords, ground_truth)
# print("Transform matrix shape:", transform.shape)
# 
# # 6. Apply affine transform to the image
# color_corrected = apply_affine_transform(hdr_image, transform)
# 
# # 7. Apply white balancing using patch 4 (index 3)
# patch4_avg = avg_rgb[3]
# white_balanced = apply_white_balance(color_corrected, patch4_avg)
# 
# # 8. Save and visualize
# save_hdr_image(color_corrected, 'color_corrected_manual.hdr')
# save_hdr_image(white_balanced, 'white_balanced_manual.hdr')
# visualize_color_correction(hdr_image, color_corrected, white_balanced) 