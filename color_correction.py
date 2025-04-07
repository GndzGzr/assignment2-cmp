import numpy as np
import matplotlib.pyplot as plt
import cv2

def crop_color_patches(image, interactive=True, saved_coordinates=None):
    """
    Crop 24 patches from color checker, either interactively or using saved coordinates.
    
    Args:
        image (numpy.ndarray): Input HDR image
        interactive (bool): Whether to use interactive selection or saved coordinates
        saved_coordinates (list): List of (x, y, size) tuples for each patch if not interactive
        
    Returns:
        tuple: (patches, coordinates) where patches is a list of 24 cropped regions and
              coordinates is a list of (x, y, size) tuples
    """
    patches = []
    coordinates = []
    
    if interactive:
        # Display the image for interactive selection
        plt.figure(figsize=(12, 8))
        plt.imshow(np.clip(image, 0, 1))  # Clip for display purposes
        plt.title("Select the center of each color patch (24 points total)")
        plt.axis('on')
        
        # Get 24 points from user interaction
        points = plt.ginput(24, timeout=0)
        plt.close()
        
        # Define patch size (you may need to adjust this)
        patch_size = min(image.shape[0], image.shape[1]) // 20
        
        # Extract patches
        for x, y in points:
            x, y = int(x), int(y)
            half_size = patch_size // 2
            
            # Ensure the patch is within image boundaries
            x_start = max(0, x - half_size)
            y_start = max(0, y - half_size)
            x_end = min(image.shape[1], x + half_size)
            y_end = min(image.shape[0], y + half_size)
            
            patch = image[y_start:y_end, x_start:x_end]
            patches.append(patch)
            coordinates.append((x, y, patch_size))
    else:
        # Use saved coordinates
        if saved_coordinates is None or len(saved_coordinates) != 24:
            raise ValueError("Need exactly 24 saved coordinates")
        
        for x, y, size in saved_coordinates:
            half_size = size // 2
            x_start = max(0, x - half_size)
            y_start = max(0, y - half_size)
            x_end = min(image.shape[1], x + half_size)
            y_end = min(image.shape[0], y + half_size)
            
            patch = image[y_start:y_end, x_start:x_end]
            patches.append(patch)
            coordinates.append((x, y, size))
    
    return patches, coordinates

def compute_average_rgb(patches):
    """
    Compute average RGB values for each patch.
    
    Args:
        patches (list): List of 24 cropped patch regions
        
    Returns:
        numpy.ndarray: 24x3 array of average RGB values
    """
    avg_rgb = []
    
    for patch in patches:
        # Compute average RGB for each patch
        avg_color = np.mean(patch, axis=(0, 1))
        avg_rgb.append(avg_color)
    
    return np.array(avg_rgb)

def convert_to_homogeneous(rgb_values):
    """
    Convert RGB values to homogeneous coordinates by appending 1.
    
    Args:
        rgb_values (numpy.ndarray): Nx3 array of RGB values
        
    Returns:
        numpy.ndarray: Nx4 array of homogeneous coordinates
    """
    # Add a column of ones to make homogeneous coordinates
    ones = np.ones((rgb_values.shape[0], 1))
    return np.hstack((rgb_values, ones))

def compute_affine_transform(source_coords, target_coords):
    """
    Compute affine transformation matrix using least squares.
    
    Args:
        source_coords (numpy.ndarray): Nx4 homogeneous coordinates of measured values
        target_coords (numpy.ndarray): Nx3 ground truth RGB values
        
    Returns:
        numpy.ndarray: 4x3 affine transformation matrix
    """
    # Solve for each channel separately
    transformation = np.zeros((4, 3))
    
    for i in range(3):  # For R, G, B channels
        # Solve least squares: source_coords * X = target_coords[:, i]
        x, residuals, rank, s = np.linalg.lstsq(source_coords, target_coords[:, i], rcond=None)
        transformation[:, i] = x
    
    return transformation

def apply_affine_transform(image, transform):
    """
    Apply affine transformation to an HDR image.
    
    Args:
        image (numpy.ndarray): Input HDR image
        transform (numpy.ndarray): 4x3 affine transformation matrix
        
    Returns:
        numpy.ndarray: Color-corrected HDR image
    """
    # Reshape image to 2D array of pixels
    h, w, c = image.shape
    pixels = image.reshape(-1, c)
    
    # Convert to homogeneous coordinates
    homogeneous = np.hstack((pixels, np.ones((pixels.shape[0], 1))))
    
    # Apply transformation
    corrected = np.dot(homogeneous, transform)
    
    # Reshape back to image
    corrected_image = corrected.reshape(h, w, c)
    
    # Clip negative values
    corrected_image = np.maximum(corrected_image, 0)
    
    return corrected_image

def apply_white_balance(image, patch4_avg):
    """
    Apply white balancing transform so that the RGB coordinates of patch 4 are equal.
    
    Args:
        image (numpy.ndarray): Input HDR image
        patch4_avg (numpy.ndarray): Average RGB value of patch 4 (white patch)
        
    Returns:
        numpy.ndarray: White-balanced HDR image
    """
    # Get the maximum value across channels of patch 4
    max_val = np.max(patch4_avg)
    
    # Compute scaling factors to make R=G=B for patch 4
    scaling = max_val / patch4_avg
    
    # Apply scaling to entire image
    balanced_image = np.zeros_like(image)
    for i in range(3):
        balanced_image[:, :, i] = image[:, :, i] * scaling[i]
    
    return balanced_image

def save_hdr_image(image, filename):
    """
    Save an HDR image to an HDR file.
    
    Args:
        image (numpy.ndarray): HDR image to save
        filename (str): Output filename
    """
    # OpenCV expects BGR order for HDR files
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, bgr_image)

def color_correct_hdr_image(hdr_image, color_checker_values, interactive=True, saved_coordinates=None):
    """
    Complete color correction and white balancing pipeline.
    
    Args:
        hdr_image (numpy.ndarray): Input HDR image
        color_checker_values (numpy.ndarray): 4x6x3 ground truth values from color checker
        interactive (bool): Whether to select patches interactively
        saved_coordinates (list): Previously saved coordinates if not interactive
        
    Returns:
        tuple: (color_corrected_image, white_balanced_image, coordinates)
    """
    # Reshape color checker values to 24x3
    ground_truth = color_checker_values.reshape(24, 3)
    
    # 1. Crop color patches
    patches, coordinates = crop_color_patches(hdr_image, interactive, saved_coordinates)
    
    # 2. Compute average RGB for each patch
    avg_rgb = compute_average_rgb(patches)
    
    # 3. Convert to homogeneous coordinates
    homogeneous_coords = convert_to_homogeneous(avg_rgb)
    
    # 4. Compute affine transformation
    transform = compute_affine_transform(homogeneous_coords, ground_truth)
    
    # 5. Apply affine transformation
    color_corrected = apply_affine_transform(hdr_image, transform)
    
    # 6. Apply white balancing (patch 4 is the 4th patch in the first row, index 3)
    patch4_avg = avg_rgb[3]
    white_balanced = apply_white_balance(color_corrected, patch4_avg)
    
    return color_corrected, white_balanced, coordinates

def visualize_color_correction(original, color_corrected, white_balanced):
    """
    Visualize original, color-corrected, and white-balanced images for comparison.
    
    Args:
        original (numpy.ndarray): Original HDR image
        color_corrected (numpy.ndarray): Color-corrected HDR image
        white_balanced (numpy.ndarray): White-balanced HDR image
    """
    # Display function for HDR images
    def display_hdr(img):
        # Simple tone mapping for display
        gamma = 2.2
        img_display = np.clip(img, 0, None)
        img_display = (img_display / (np.max(img_display) + 1e-6))**(1/gamma)
        return np.clip(img_display, 0, 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(display_hdr(original))
    axes[0].set_title('Original HDR')
    axes[0].axis('off')
    
    axes[1].imshow(display_hdr(color_corrected))
    axes[1].set_title('Color Corrected')
    axes[1].axis('off')
    
    axes[2].imshow(display_hdr(white_balanced))
    axes[2].set_title('White Balanced')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show() 