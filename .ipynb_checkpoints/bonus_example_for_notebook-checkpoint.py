# Copy and paste this code into your Jupyter notebook

# First, import the functions from bonus_functions.py
from bonus_functions import (load_exposure_stack, preprocess_raw_images, 
                            preprocess_jpeg_images, create_hdr_image,
                            save_hdr_image, apply_tone_mapping, 
                            experiment_with_tone_mapping, 
                            find_best_tone_mapping_params, 
                            compare_raw_jpeg_hdr)

import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
os.makedirs('bonus_results', exist_ok=True)

# Step 1: Load your exposure stacks
# Update these paths to point to your exposure stacks
raw_directory = 'path/to/your/raw/photos'  # Directory with your RAW exposure stack
jpeg_directory = 'path/to/your/jpeg/photos'  # Directory with your JPEG exposure stack

# Load RAW images (uncomment when ready)
# raw_images, raw_exposures = load_exposure_stack(raw_directory, 'RAW')
# print(f"Loaded {len(raw_images)} RAW images with exposure times: {raw_exposures}")

# Load JPEG images (uncomment when ready)
# jpeg_images, jpeg_exposures = load_exposure_stack(jpeg_directory, 'JPEG')
# print(f"Loaded {len(jpeg_images)} JPEG images with exposure times: {jpeg_exposures}")

# Step 2: Preprocess images

# Preprocess RAW images (uncomment when ready)
# processed_raw = preprocess_raw_images(raw_images)
# print("RAW images preprocessed")

# Preprocess JPEG images (uncomment when ready)
# processed_jpeg, response_curve = preprocess_jpeg_images(jpeg_images, jpeg_exposures)
# print("JPEG images preprocessed and linearized")

# Plot the recovered response curve from JPEGs
# plt.figure(figsize=(10, 6))
# plt.plot(range(256), response_curve)
# plt.title("Recovered Camera Response Function")
# plt.xlabel("Pixel Value")
# plt.ylabel("Log Exposure")
# plt.grid(True)
# plt.savefig('bonus_results/response_curve.png')
# plt.show()

# Step 3: Create HDR images
# NOTE: For testing or if you don't have actual images,
# you can use dummy data:
dummy_raw = [np.random.rand(100, 100, 3) for _ in range(3)]
dummy_jpeg = [np.random.rand(100, 100, 3) for _ in range(3)]
dummy_exposures = [1/4, 1/2, 1]

# Create HDR images - use dummy data for testing, replace with real data later
raw_hdr = create_hdr_image(dummy_raw, dummy_exposures, 'gaussian', 'logarithmic')
jpeg_hdr = create_hdr_image(dummy_jpeg, dummy_exposures, 'gaussian', 'logarithmic')

# Uncomment for real data:
# raw_hdr = create_hdr_image(processed_raw, raw_exposures, 'gaussian', 'logarithmic')
# jpeg_hdr = create_hdr_image(processed_jpeg, jpeg_exposures, 'gaussian', 'logarithmic')

print("HDR images created")

# Step 4: Save HDR images
save_hdr_image(raw_hdr, 'bonus_results/raw_hdr.hdr')
save_hdr_image(jpeg_hdr, 'bonus_results/jpeg_hdr.hdr')

# Step 5: Compare RAW and JPEG HDR images
compare_raw_jpeg_hdr(raw_hdr, jpeg_hdr, 'bonus_results/hdr_comparison.png')

# Step 6: Experiment with different tone mapping parameters
print("\nExperimenting with tone mapping parameters for RAW HDR...")
experiment_with_tone_mapping(raw_hdr, 'bonus_results/raw_tone_mapping')

print("\nExperimenting with tone mapping parameters for JPEG HDR...")
experiment_with_tone_mapping(jpeg_hdr, 'bonus_results/jpeg_tone_mapping')

# Step 7: Let user find the best parameters interactively
print("\nFinding best tone mapping parameters for RAW HDR...")
raw_key, raw_burn, raw_gamma = find_best_tone_mapping_params(raw_hdr)

print("\nFinding best tone mapping parameters for JPEG HDR...")
jpeg_key, jpeg_burn, jpeg_gamma = find_best_tone_mapping_params(jpeg_hdr)

# Step 8: Apply final tone mapping with best parameters
final_raw_ldr = apply_tone_mapping(raw_hdr, 'luminance', raw_key, raw_burn, raw_gamma)
final_jpeg_ldr = apply_tone_mapping(jpeg_hdr, 'luminance', jpeg_key, jpeg_burn, raw_gamma)

# Step 9: Save final tone-mapped images
plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
plt.imshow(final_raw_ldr)
plt.title(f'RAW HDR Tone Mapped (Key={raw_key}, Burn={raw_burn}, Gamma={raw_gamma})')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(final_jpeg_ldr)
plt.title(f'JPEG HDR Tone Mapped (Key={jpeg_key}, Burn={jpeg_burn}, Gamma={jpeg_gamma})')
plt.axis('off')

plt.tight_layout()
plt.savefig('bonus_results/final_comparison.png')
plt.show()

# Save individual images
plt.imsave('bonus_results/final_raw_tonemap.png', final_raw_ldr)
plt.imsave('bonus_results/final_jpeg_tonemap.png', final_jpeg_ldr)

print("\nBonus task complete! All results saved to 'bonus_results' directory.")

# EXTRA: Testing a range of Key and Burn values to find optimal parameters
def create_parameter_grid(hdr_image, method='luminance'):
    # Parameters to test
    key_values = [0.05, 0.1, 0.18, 0.25, 0.36]
    burn_values = [0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Create grid figure
    plt.figure(figsize=(15, 12))
    
    for i, key in enumerate(key_values):
        for j, burn in enumerate(burn_values):
            # Apply tone mapping
            ldr = apply_tone_mapping(hdr_image, method, key, burn)
            
            # Display
            plt.subplot(len(key_values), len(burn_values), i*len(burn_values) + j + 1)
            plt.imshow(ldr)
            plt.title(f'Key={key}, Burn={burn}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('bonus_results/parameter_grid.png', dpi=300)
    plt.show()

# Uncomment to run parameter grid search
# create_parameter_grid(raw_hdr) 