import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
from weighting_schemes import WeightingSchemes
from hdr_merging import HDRMerger
from response_calibration import ResponseCalibration
from tone_mapping import ToneMapper

def load_exposure_stack(directory, file_format):
    """
    Load an exposure stack of images from a directory.
    
    Args:
        directory (str): Directory containing exposure stack images
        file_format (str): File format ('RAW', 'JPEG', 'NEF', 'JPG', etc.)
        
    Returns:
        tuple: (images, exposure_times) where images is a list of image arrays
               and exposure_times is a list of corresponding exposure times
    """
    # Get image file paths
    if file_format.upper() == 'RAW' or file_format.upper() == 'NEF':
        pattern = os.path.join(directory, f'*.NEF')
    else:  # JPEG/JPG
        pattern = os.path.join(directory, f'*.jpg')
        if not glob.glob(pattern):
            pattern = os.path.join(directory, f'*.jpeg')
    
    # Get sorted list of files
    file_paths = sorted(glob.glob(pattern))
    
    if len(file_paths) == 0:
        raise ValueError(f"No {file_format} files found in {directory}")
    
    print(f"Loading {len(file_paths)} {file_format} images...")
    
    # Load images
    images = []
    for file_path in file_paths:
        print(f"  Loading {os.path.basename(file_path)}")
        if file_format.upper() == 'RAW' or file_format.upper() == 'NEF':
            # For RAW files, use cv2.imread with ANYDEPTH flag
            img = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        else:
            # For JPEG/JPG files
            img = cv2.imread(file_path)
        
        # Convert BGR to RGB
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        # Add to list
        images.append(img)
    
    # Try to extract exposure times from EXIF data
    # For simplicity, we'll use a doubling sequence if EXIF data is not available
    try:
        import exifread
        exposure_times = []
        for file_path in file_paths:
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f)
                if 'EXIF ExposureTime' in tags:
                    exp_time = tags['EXIF ExposureTime'].values[0]
                    # Convert fraction to float
                    if isinstance(exp_time, exifread.utils.Ratio):
                        exp_time = float(exp_time.num) / float(exp_time.den)
                    else:
                        exp_time = float(exp_time)
                    exposure_times.append(exp_time)
                else:
                    # Fall back to doubling sequence
                    raise ValueError("ExposureTime tag not found")
    except (ImportError, ValueError):
        # If exifread not installed or EXIF data not available, use doubling sequence
        print("Could not extract exposure times from EXIF data. Using doubling sequence...")
        exposure_times = [2**i for i in range(len(images))]
    
    print(f"Exposure times: {exposure_times}")
    
    return images, exposure_times

def preprocess_raw_images(raw_images):
    """
    Preprocess RAW images for HDR merging (normalization, etc.).
    
    Args:
        raw_images (list): List of RAW image arrays
        
    Returns:
        list: Preprocessed images ready for HDR merging
    """
    processed_images = []
    
    for img in raw_images:
        # Convert to float32
        img_float = img.astype(np.float32)
        
        # Normalize to [0, 1] range
        if img_float.max() > 0:
            img_float = img_float / img_float.max()
        
        processed_images.append(img_float)
    
    return processed_images

def preprocess_jpeg_images(jpeg_images, exposure_times):
    """
    Preprocess JPEG images for HDR merging (linearization, etc.).
    
    Args:
        jpeg_images (list): List of JPEG image arrays
        exposure_times (list): List of exposure times
        
    Returns:
        list: Preprocessed (linearized) images ready for HDR merging
    """
    # Normalize JPEG images to [0, 1]
    normalized_images = [img.astype(np.float32) / 255.0 for img in jpeg_images]
    
    # Recover camera response function
    calibration = ResponseCalibration(normalized_images, exposure_times)
    g = calibration.solve_g(n_samples=100)
    
    # Linearize images
    linearized_images = []
    for img in normalized_images:
        # Convert to uint8 for indexing into g
        img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        linear_img = calibration.linearize_image(img_uint8, g)
        linearized_images.append(linear_img)
    
    return linearized_images, g

def create_hdr_image(images, exposure_times, weighting_method='gaussian', merging_method='logarithmic'):
    """
    Create an HDR image from an exposure stack.
    
    Args:
        images (list): List of preprocessed images
        exposure_times (list): List of exposure times
        weighting_method (str): Weighting method ('uniform', 'tent', 'gaussian', 'photon')
        merging_method (str): Merging method ('linear', 'logarithmic')
        
    Returns:
        numpy.ndarray: HDR image
    """
    # Create HDR merger
    merger = HDRMerger(weighting_method=weighting_method)
    
    # Merge images using specified method
    if merging_method == 'linear':
        hdr_image = merger.linear_merging(images, exposure_times)
    else:
        hdr_image = merger.logarithmic_merging(images, exposure_times)
    
    return hdr_image

def save_hdr_image(hdr_image, filename):
    """
    Save HDR image to file.
    
    Args:
        hdr_image (numpy.ndarray): HDR image
        filename (str): Output filename
    """
    # Convert RGB to BGR for OpenCV
    bgr_image = cv2.cvtColor(hdr_image, cv2.COLOR_RGB2BGR)
    
    # Save as HDR file
    cv2.imwrite(filename, bgr_image)
    print(f"Saved HDR image to {filename}")

def apply_tone_mapping(hdr_image, method='luminance', key=0.18, burn=0.85, gamma=2.2):
    """
    Apply tone mapping to HDR image.
    
    Args:
        hdr_image (numpy.ndarray): HDR image
        method (str): Tone mapping method ('rgb' or 'luminance')
        key (float): Key value (brightness)
        burn (float): Burn value (highlight compression)
        gamma (float): Gamma value for gamma correction
        
    Returns:
        numpy.ndarray: Tone mapped LDR image
    """
    # Create tone mapper with specified parameters
    tone_mapper = ToneMapper(key=key, burn=burn)
    
    # Apply tone mapping
    ldr_image = tone_mapper.tone_map(hdr_image, method=method, apply_gamma_correction=True)
    
    return ldr_image

def experiment_with_tone_mapping(hdr_image, output_dir='tone_mapping_results'):
    """
    Experiment with different tone mapping parameters and save results.
    
    Args:
        hdr_image (numpy.ndarray): HDR image
        output_dir (str): Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create tone mapper
    tone_mapper = ToneMapper()
    
    # Parameters to experiment with
    methods = ['luminance', 'rgb']
    key_values = [0.09, 0.18, 0.36]
    burn_values = [0.7, 0.85, 1.0]
    
    # Create a figure to display results
    fig = plt.figure(figsize=(15, 10))
    
    idx = 1
    for method in methods:
        for key in key_values:
            for burn in burn_values:
                # Set parameters
                tone_mapper.key = key
                tone_mapper.burn = burn
                
                # Apply tone mapping
                ldr_image = tone_mapper.tone_map(hdr_image, method=method)
                
                # Save image
                filename = f"{output_dir}/tonemap_{method}_key{key}_burn{burn}.png"
                tone_mapper.save_tone_mapped_image(ldr_image, filename)
                
                # Add to figure
                plt.subplot(len(methods), len(key_values) * len(burn_values), idx)
                plt.imshow(ldr_image)
                plt.title(f"{method}, key={key}, burn={burn}")
                plt.axis('off')
                idx += 1
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tone_mapping_comparison.png", dpi=300)
    plt.show()

def find_best_tone_mapping_params(hdr_image, method='luminance'):
    """
    Interactive tool to find the best tone mapping parameters for an HDR image.
    
    Args:
        hdr_image (numpy.ndarray): HDR image
        method (str): Tone mapping method ('rgb' or 'luminance')
        
    Returns:
        tuple: Best parameters (key, burn, gamma)
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create tone mapper
    tone_mapper = ToneMapper()
    
    # Initial parameters
    key = 0.18
    burn = 0.85
    gamma = 2.2
    
    # Apply tone mapping with initial parameters
    ldr_image = tone_mapper.tone_map(hdr_image, method=method)
    
    # Display image
    plt.imshow(ldr_image)
    plt.title(f"Method: {method}, Key: {key:.2f}, Burn: {burn:.2f}, Gamma: {gamma:.2f}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Get user input for parameters
    print("\nEnter parameters (press Enter to keep current value):")
    
    try:
        key_input = input(f"Key (current={key:.2f}): ")
        if key_input:
            key = float(key_input)
        
        burn_input = input(f"Burn (current={burn:.2f}): ")
        if burn_input:
            burn = float(burn_input)
        
        gamma_input = input(f"Gamma (current={gamma:.2f}): ")
        if gamma_input:
            gamma = float(gamma_input)
    except ValueError:
        print("Invalid input, using current values.")
    
    # Update tone mapper with new parameters
    tone_mapper.key = key
    tone_mapper.burn = burn
    
    # Apply tone mapping with new parameters
    ldr_image = tone_mapper.tone_map(hdr_image, method=method)
    
    # Display final image
    plt.figure(figsize=(12, 8))
    plt.imshow(ldr_image)
    plt.title(f"Method: {method}, Key: {key:.2f}, Burn: {burn:.2f}, Gamma: {gamma:.2f}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print(f"Final parameters: Key={key:.2f}, Burn={burn:.2f}, Gamma={gamma:.2f}")
    
    return key, burn, gamma

def compare_raw_jpeg_hdr(raw_hdr, jpeg_hdr, output_file='hdr_comparison.png'):
    """
    Compare HDR images created from RAW and JPEG.
    
    Args:
        raw_hdr (numpy.ndarray): HDR image from RAW
        jpeg_hdr (numpy.ndarray): HDR image from JPEG
        output_file (str): Output filename for comparison image
    """
    # Create tone mapper for display
    tone_mapper = ToneMapper()
    
    # Tone map both images with the same parameters
    raw_ldr = tone_mapper.tone_map(raw_hdr)
    jpeg_ldr = tone_mapper.tone_map(jpeg_hdr)
    
    # Create figure for comparison
    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 2, 1)
    plt.imshow(raw_ldr)
    plt.title('HDR from RAW')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(jpeg_ldr)
    plt.title('HDR from JPEG')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()

def bonus_workflow_example():
    """
    Example workflow for the bonus task.
    """
    # 1. Load exposure stack
    print("Step 1: Load exposure stacks")
    raw_images, raw_exposures = load_exposure_stack('your_raw_directory', 'RAW')
    jpeg_images, jpeg_exposures = load_exposure_stack('your_jpeg_directory', 'JPEG')
    
    # 2. Preprocess images
    print("\nStep 2: Preprocess images")
    processed_raw = preprocess_raw_images(raw_images)
    processed_jpeg, response_curve = preprocess_jpeg_images(jpeg_images, jpeg_exposures)
    
    # 3. Create HDR images
    print("\nStep 3: Create HDR images")
    raw_hdr = create_hdr_image(processed_raw, raw_exposures, 'gaussian', 'logarithmic')
    jpeg_hdr = create_hdr_image(processed_jpeg, jpeg_exposures, 'gaussian', 'logarithmic')
    
    # 4. Save HDR images
    print("\nStep 4: Save HDR images")
    save_hdr_image(raw_hdr, 'raw_hdr.hdr')
    save_hdr_image(jpeg_hdr, 'jpeg_hdr.hdr')
    
    # 5. Compare HDR images
    print("\nStep 5: Compare HDR images")
    compare_raw_jpeg_hdr(raw_hdr, jpeg_hdr)
    
    # 6. Experiment with tone mapping
    print("\nStep 6: Experiment with tone mapping")
    experiment_with_tone_mapping(raw_hdr, 'raw_tone_mapping')
    experiment_with_tone_mapping(jpeg_hdr, 'jpeg_tone_mapping')
    
    # 7. Find best tone mapping parameters
    print("\nStep 7: Find best tone mapping parameters for RAW HDR")
    raw_key, raw_burn, raw_gamma = find_best_tone_mapping_params(raw_hdr)
    
    print("\nStep 8: Find best tone mapping parameters for JPEG HDR")
    jpeg_key, jpeg_burn, jpeg_gamma = find_best_tone_mapping_params(jpeg_hdr)
    
    # 9. Create final tone-mapped images with best parameters
    print("\nStep 9: Create final tone-mapped images")
    final_raw_ldr = apply_tone_mapping(raw_hdr, 'luminance', raw_key, raw_burn, raw_gamma)
    final_jpeg_ldr = apply_tone_mapping(jpeg_hdr, 'luminance', jpeg_key, jpeg_burn, jpeg_gamma)
    
    # 10. Save final images
    print("\nStep 10: Save final images")
    cv2.imwrite('final_raw_tonemap.png', cv2.cvtColor((final_raw_ldr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite('final_jpeg_tonemap.png', cv2.cvtColor((final_jpeg_ldr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    print("\nWorkflow complete!") 