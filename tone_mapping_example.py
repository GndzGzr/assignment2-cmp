import numpy as np
import cv2
import matplotlib.pyplot as plt
from tone_mapping import ToneMapper

# Example usage code to copy into your Jupyter notebook

# Step 1: Load your HDR image
# Replace with your HDR image path
hdr_image_path = 'white_balanced.hdr'  # Use the white-balanced HDR image from previous step
hdr_image = cv2.imread(hdr_image_path, cv2.IMREAD_ANYDEPTH)
hdr_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Step 2: Create a ToneMapper instance with default parameters
tone_mapper = ToneMapper(key=0.18, burn=0.85, epsilon=1e-6)

# Step 3: Apply tone mapping using the luminance method (default)
ldr_image = tone_mapper.tone_map(hdr_image, method='luminance', apply_gamma_correction=True)

# Step 4: Save the tone mapped image
tone_mapper.save_tone_mapped_image(ldr_image, 'tone_mapped_luminance.png')

# Step 5: Apply tone mapping using the RGB method
ldr_image_rgb = tone_mapper.tone_map(hdr_image, method='rgb', apply_gamma_correction=True)

# Step 6: Save the RGB tone mapped image
tone_mapper.save_tone_mapped_image(ldr_image_rgb, 'tone_mapped_rgb.png')

# Step 7: Compare the two methods
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(ldr_image)
plt.title('Luminance-based Tone Mapping')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(ldr_image_rgb)
plt.title('RGB-based Tone Mapping')
plt.axis('off')

plt.tight_layout()
plt.savefig('tone_mapping_comparison.png')
plt.show()

# Step 8: Explore different parameter settings using the visualization function
# This will create a grid of images with different key and gamma values
tone_mapper.visualize_tone_mapping(
    hdr_image,
    gamma_values=[1.0, 2.2, 3.0],
    key_values=[0.09, 0.18, 0.36],
    method='luminance'
)

# Example of exploring different burn values
plt.figure(figsize=(15, 5))

burn_values = [0.6, 0.85, 1.0]
for i, burn in enumerate(burn_values):
    # Update burn parameter
    tone_mapper.burn = burn
    tone_mapper.key = 0.18  # Reset key to default
    
    # Apply tone mapping
    ldr = tone_mapper.tone_map(hdr_image, method='luminance')
    
    # Display
    plt.subplot(1, 3, i+1)
    plt.imshow(ldr)
    plt.title(f'Burn: {burn}')
    plt.axis('off')

plt.tight_layout()
plt.savefig('burn_comparison.png')
plt.show()

# Create a custom function to find the best parameters
def find_best_parameters(hdr_image, tone_mapper, key_range=[0.05, 0.1, 0.18, 0.25, 0.36], 
                         burn_range=[0.6, 0.75, 0.85, 0.95, 1.0]):
    """
    Test different parameter combinations to find the best tone mapping parameters.
    
    Args:
        hdr_image (numpy.ndarray): Input HDR image
        tone_mapper (ToneMapper): Tone mapper instance
        key_range (list): Range of key values to try
        burn_range (list): Range of burn values to try
        
    Returns:
        tuple: Best parameters (key, burn) and corresponding tone mapped image
    """
    plt.figure(figsize=(len(burn_range)*3, len(key_range)*3))
    
    for i, key in enumerate(key_range):
        for j, burn in enumerate(burn_range):
            # Set parameters
            tone_mapper.key = key
            tone_mapper.burn = burn
            
            # Apply tone mapping
            ldr = tone_mapper.tone_map(hdr_image, method='luminance')
            
            # Display
            plt.subplot(len(key_range), len(burn_range), i*len(burn_range) + j + 1)
            plt.imshow(ldr)
            plt.title(f'Key: {key}, Burn: {burn}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('parameter_exploration.png')
    plt.show()

# Test with different parameters
find_best_parameters(hdr_image, tone_mapper)

# After exploring parameters, you can save the final tone mapped image
# with your preferred settings
tone_mapper.key = 0.18  # Set your preferred key value
tone_mapper.burn = 0.85  # Set your preferred burn value
final_image = tone_mapper.tone_map(hdr_image, method='luminance')
tone_mapper.save_tone_mapped_image(final_image, 'final_tone_mapped.png') 