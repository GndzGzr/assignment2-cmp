import numpy as np
from weighting_schemes import WeightingSchemes

class HDRMerger:
    def __init__(self, weighting_method='gaussian', z_min=0.05, z_max=0.95):
        """
        Initialize the HDR merger.
        
        Args:
            weighting_method (str): Weighting method to use ('uniform', 'tent', 'gaussian', 'photon')
            z_min (float): Minimum intensity value (default: 0.05)
            z_max (float): Maximum intensity value (default: 0.95)
        """
        self.weights = WeightingSchemes(z_min, z_max)
        self.weighting_method = getattr(self.weights, weighting_method)
        
    def linear_merging(self, images, exposure_times):
        """
        Merge multiple LDR linear images into an HDR image using linear merging.
        
        Args:
            images (list): List of linearized LDR images
            exposure_times (list): List of exposure times for each image
            
        Returns:
            numpy.ndarray: HDR image
        """
        # Initialize the HDR image with zeros
        hdr_image = np.zeros_like(images[0], dtype=np.float32)
        
        # Initialize weight sum array to handle division
        weight_sum = np.zeros_like(images[0], dtype=np.float32)
        
        # For each image in the stack
        for img, t in zip(images, exposure_times):
            # Calculate weights based on original pixel values (before linearization)
            original_values = np.clip(img * 255, 0, 255).astype(np.uint8)
            weights = self.weighting_method(original_values / 255.0)
            
            # Add weighted contribution to HDR image
            hdr_image += weights * img / t
            weight_sum += weights
        
        # Handle division by zero (pixels with no valid exposure)
        # For over-exposed pixels, assign maximum valid pixel value
        # For under-exposed pixels, assign minimum valid pixel value
        valid_mask = weight_sum > 0
        hdr_image[valid_mask] /= weight_sum[valid_mask]
        
        # Find over-exposed and under-exposed pixels
        # This is a simple approach, could be refined based on specific needs
        over_exposed = (~valid_mask) & (np.mean(images[-1], axis=-1, keepdims=True) > 0.95)
        under_exposed = (~valid_mask) & (~over_exposed)
        
        # Assign values to over/under exposed pixels
        max_valid_value = np.max(hdr_image[valid_mask]) if np.any(valid_mask) else 1.0
        min_valid_value = np.min(hdr_image[valid_mask]) if np.any(valid_mask) else 0.0
        
        hdr_image[over_exposed] = max_valid_value
        hdr_image[under_exposed] = min_valid_value
        
        return hdr_image
    
    def logarithmic_merging(self, images, exposure_times, epsilon=1e-6):
        """
        Merge multiple LDR linear images into an HDR image using logarithmic merging.
        
        Args:
            images (list): List of linearized LDR images
            exposure_times (list): List of exposure times for each image
            epsilon (float): Small constant to avoid singularity in log
            
        Returns:
            numpy.ndarray: HDR image
        """
        # Initialize numerator and denominator for the weighted sum
        numerator = np.zeros_like(images[0], dtype=np.float32)
        denominator = np.zeros_like(images[0], dtype=np.float32)
        
        # For each image in the stack
        for img, t in zip(images, exposure_times):
            # Calculate weights based on original pixel values (before linearization)
            original_values = np.clip(img * 255, 0, 255).astype(np.uint8)
            weights = self.weighting_method(original_values / 255.0)
            
            # Add weighted contribution to log HDR image
            # Add epsilon to avoid log(0)
            valid_pixels = img > 0
            log_img = np.zeros_like(img, dtype=np.float32)
            log_img[valid_pixels] = np.log(img[valid_pixels] + epsilon) - np.log(t)
            
            numerator += weights * log_img
            denominator += weights
        
        # Handle division by zero (pixels with no valid exposure)
        valid_mask = denominator > 0
        hdr_log = np.zeros_like(numerator)
        hdr_log[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
        
        # Find over-exposed and under-exposed pixels
        over_exposed = (~valid_mask) & (np.mean(images[-1], axis=-1, keepdims=True) > 0.95)
        under_exposed = (~valid_mask) & (~over_exposed)
        
        # Assign values to over/under exposed pixels
        max_valid_value = np.max(hdr_log[valid_mask]) if np.any(valid_mask) else 0.0
        min_valid_value = np.min(hdr_log[valid_mask]) if np.any(valid_mask) else -10.0
        
        hdr_log[over_exposed] = max_valid_value
        hdr_log[under_exposed] = min_valid_value
        
        # Convert from log domain back to linear
        hdr_image = np.exp(hdr_log)
        
        return hdr_image 