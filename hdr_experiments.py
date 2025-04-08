import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
from weighting_schemes import WeightingSchemes
from response_calibration import ResponseCalibration
from hdr_merging import HDRMerger
import time
import glob

def load_images(image_dir, image_format):
    """
    Load all images from a directory with a specific format (jpg or tiff).
    
    Args:
        image_dir (str): Directory containing images
        image_format (str): 'jpg' for rendered or 'tiff' for RAW
        
    Returns:
        tuple: (list of images, list of exposure times)
    """
    # Pattern for filenames
    pattern = os.path.join(image_dir, f'*.{image_format}')
    
    # Get list of files sorted by name
    file_list = sorted(glob.glob(pattern))
    
    if len(file_list) == 0:
        raise ValueError(f"No {image_format} files found in {image_dir}")
    
    images = []
    
    print(f"Loading {len(file_list)} {image_format} images...")
    
    for file_path in file_list:
        print(f"Loading {os.path.basename(file_path)}")
        if image_format.lower() == 'jpg':
            # Load JPG images with cv2
            img = cv2.imread(file_path)
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # Load TIFF images with cv2
            img = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            # Convert BGR to RGB if needed
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize image values
        if image_format.lower() == 'jpg':
            img = img.astype(np.float32) / 255.0
        else:
            # RAW TIFF values are already linear, normalize to [0, 1]
            img_min = np.min(img)
            img_max = np.max(img)
            img = (img.astype(np.float32) - img_min) / (img_max - img_min)
            
        images.append(img)
    
    # For simplicity, simulate exposure times based on image index
    # In a real scenario, these would be extracted from EXIF data
    # Here we assume each image is taken with doubling exposure time 
    exposure_times = [2**i for i in range(len(images))]
    
    return images, exposure_times

def downscale_images(images, scale_factor=0.25):
    """
    Downscale images for faster processing and display.
    
    Args:
        images (list): List of images
        scale_factor (float): Scale factor to resize images
        
    Returns:
        list: Downscaled images
    """
    downscaled_images = []
    for img in images:
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        downscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        downscaled_images.append(downscaled)
    return downscaled_images

def normalize_for_display(hdr_image):
    """
    Apply a simple normalization for displaying HDR images.
    
    Args:
        hdr_image (numpy.ndarray): HDR image
        
    Returns:
        numpy.ndarray: Image normalized for display
    """
    # Apply a simple tone mapping using log scaling
    epsilon = 1e-6
    log_img = np.log(hdr_image + epsilon)
    
    # Normalize to [0, 1]
    normalized = (log_img - np.min(log_img)) / (np.max(log_img) - np.min(log_img))
    
    # Gamma correction to enhance perceptual quality
    gamma = 2.2
    normalized = np.power(normalized, 1/gamma)
    
    # Ensure values are in [0, 1]
    normalized = np.clip(normalized, 0, 1)
    
    return normalized

def run_experiments():
    """
    Run experiments with different combinations of image types, merging methods, and weighting schemes.
    """
    # Parameters for experiments
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Lab_Booth")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Experiment combinations
    image_types = ["jpg", "tiff"]
    merging_methods = ["linear", "logarithmic"]
    weighting_schemes = ["uniform", "tent", "gaussian", "photon"]
    
    # Dictionary to store results
    results = {}
    
    # Load and process images
    for image_type in image_types:
        print(f"\nProcessing {image_type.upper()} images...")
        
        # Load images
        images, exposure_times = load_images(data_dir, image_type)
        
        # Downscale for faster processing
        images = downscale_images(images, scale_factor=0.1)
        
        # Linearize if JPG (rendered images)
        if image_type == "jpg":
            print("Calibrating camera response function...")
            calibration = ResponseCalibration(images, exposure_times)
            g = calibration.solve_g(n_samples=100)
            
            # Plot g function
            plt.figure(figsize=(10, 6))
            plt.plot(range(256), g)
            plt.title("Recovered Camera Response Function (g)")
            plt.xlabel("Pixel Value")
            plt.ylabel("Log Exposure")
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "response_function.png"))
            plt.close()
            
            # Linearize images
            print("Linearizing images...")
            linearized_images = []
            for img in images:
                # Convert to uint8 for indexing into g
                img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
                linear_img = calibration.linearize_image(img_uint8, g)
                linearized_images.append(linear_img)
            
            images_to_merge = linearized_images
        else:
            # TIFF images are already linear
            images_to_merge = images
        
        # Process images with different methods
        for merging_method in merging_methods:
            for weighting_scheme in weighting_schemes:
                print(f"Creating HDR image with {merging_method} merging and {weighting_scheme} weighting...")
                
                # Create HDR merger
                merger = HDRMerger(weighting_method=weighting_scheme)
                
                # Time the HDR creation
                start_time = time.time()
                
                # Create HDR image using the specified method
                if merging_method == "linear":
                    hdr_image = merger.linear_merging(images_to_merge, exposure_times)
                else:
                    hdr_image = merger.logarithmic_merging(images_to_merge, exposure_times)
                
                elapsed_time = time.time() - start_time
                
                # Normalize for display
                display_image = normalize_for_display(hdr_image)
                
                # Store result
                key = f"{image_type}_{merging_method}_{weighting_scheme}"
                results[key] = {
                    "hdr_image": hdr_image,
                    "display_image": display_image,
                    "time": elapsed_time
                }
                
                # Save display image
                plt.imsave(os.path.join(output_dir, f"{key}.png"), display_image)
    
    return results

def visualize_results(results):
    """
    Create a comparison visualization of all results.
    
    Args:
        results (dict): Dictionary containing experiment results
    """
    # Define the experiment parameters
    image_types = ["jpg", "tiff"]
    merging_methods = ["linear", "logarithmic"]
    weighting_schemes = ["uniform", "tent", "gaussian", "photon"]
    
    # Create a large figure for all combinations
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(len(image_types) * len(merging_methods), len(weighting_schemes), figure=fig)
    
    # Add title
    fig.suptitle("HDR Image Comparison across Methods", fontsize=16)
    
    # Add column headers (weighting schemes)
    for j, scheme in enumerate(weighting_schemes):
        fig.text(0.125 + j * 0.75/len(weighting_schemes), 0.95, scheme.capitalize(), 
                 ha='center', va='center', fontsize=12)
    
    # Plot each combination
    for i1, image_type in enumerate(image_types):
        for i2, merge_method in enumerate(merging_methods):
            row = i1 * len(merging_methods) + i2
            
            # Add row label
            fig.text(0.05, 0.85 - row * 0.7/len(image_types)/len(merging_methods), 
                     f"{image_type.upper()} - {merge_method.capitalize()}", 
                     ha='center', va='center', fontsize=10, rotation=90)
            
            for j, weight_method in enumerate(weighting_schemes):
                key = f"{image_type}_{merge_method}_{weight_method}"
                
                if key in results:
                    ax = fig.add_subplot(gs[row, j])
                    ax.imshow(results[key]["display_image"])
                    ax.set_title(f"Time: {results[key]['time']:.2f}s", fontsize=8)
                    ax.axis('off')
    
    plt.tight_layout(rect=[0.07, 0.03, 0.98, 0.95])
    plt.savefig("results/all_hdr_comparisons.png", dpi=300)
    plt.close()
    
    # Create a table of execution times
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    
    # Prepare data for table
    cell_data = []
    for i1, image_type in enumerate(image_types):
        for i2, merge_method in enumerate(merging_methods):
            row_data = []
            for weight_method in weighting_schemes:
                key = f"{image_type}_{merge_method}_{weight_method}"
                if key in results:
                    row_data.append(f"{results[key]['time']:.2f}s")
                else:
                    row_data.append("N/A")
            cell_data.append(row_data)
    
    # Create table
    row_labels = [f"{img_type.upper()} - {merge}" 
                 for img_type in image_types 
                 for merge in merging_methods]
    col_labels = [scheme.capitalize() for scheme in weighting_schemes]
    
    table = ax.table(cellText=cell_data,
                     rowLabels=row_labels,
                     colLabels=col_labels,
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.axis('off')
    plt.title("HDR Generation Time (seconds)")
    plt.tight_layout()
    plt.savefig("results/timing_table.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    # Run the experiments
    results = run_experiments()
    
    # Visualize the results
    visualize_results(results)
    
    print("Experiments completed. Results saved to 'results' directory.") 