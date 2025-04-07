import numpy as np
import cv2
import matplotlib.pyplot as plt

class ToneMapper:
    """
    Implementation of Reinhard et al. photographic tone mapping operator.
    """
    
    def __init__(self, key=0.18, burn=0.85, epsilon=1e-6):
        """
        Initialize the tone mapper with parameters.
        
        Args:
            key (float): Key value that determines brightness (0.18 is default)
            burn (float): Burn value for highlight compression (0.85 is default)
            epsilon (float): Small constant to avoid log(0) (1e-6 is default)
        """
        self.key = key
        self.burn = burn
        self.epsilon = epsilon
    
    def rgb_to_xyz(self, rgb_image):
        """
        Convert RGB to XYZ color space.
        
        Args:
            rgb_image (numpy.ndarray): RGB image
            
        Returns:
            numpy.ndarray: Y channel (luminance)
        """
        # RGB to XYZ conversion matrix
        conversion_matrix = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        
        # Reshape image for matrix multiplication
        h, w, c = rgb_image.shape
        rgb_flat = rgb_image.reshape(-1, 3)
        
        # Convert to XYZ
        xyz_flat = np.dot(rgb_flat, conversion_matrix.T)
        xyz = xyz_flat.reshape(h, w, 3)
        
        # Return Y channel (luminance)
        return xyz[:, :, 1]
    
    def log_average_luminance(self, image, use_rgb=False):
        """
        Calculate log average luminance of the image.
        
        Args:
            image (numpy.ndarray): Input HDR image
            use_rgb (bool): If True, compute for each RGB channel separately,
                            otherwise compute for luminance
            
        Returns:
            float or numpy.ndarray: Log average luminance (scalar or per-channel)
        """
        if use_rgb:
            # Calculate for each channel separately
            log_avg = np.zeros(3)
            
            for i in range(3):
                channel = image[:, :, i]
                log_avg[i] = np.exp(np.mean(np.log(channel + self.epsilon)))
            
            return log_avg
        else:
            # Calculate luminance
            luminance = self.rgb_to_xyz(image)
            
            # Calculate log average
            return np.exp(np.mean(np.log(luminance + self.epsilon)))
    
    def tone_map_rgb(self, hdr_image):
        """
        Apply tone mapping to each RGB channel separately.
        
        Args:
            hdr_image (numpy.ndarray): Input HDR image
            
        Returns:
            numpy.ndarray: Tone mapped LDR image
        """
        # Calculate log average per channel
        l_avg = self.log_average_luminance(hdr_image, use_rgb=True)
        
        # Scale image by key value / log average
        scaled = np.zeros_like(hdr_image)
        for i in range(3):
            scaled[:, :, i] = (self.key / l_avg[i]) * hdr_image[:, :, i]
        
        # Calculate white level (per channel)
        white_levels = self.burn * np.array([
            np.max(hdr_image[:, :, 0]),
            np.max(hdr_image[:, :, 1]),
            np.max(hdr_image[:, :, 2])
        ])
        
        # Apply tone mapping operator
        ldr_image = np.zeros_like(hdr_image)
        for i in range(3):
            # Apply Reinhard mapping: L(x) = L(x) * (1 + L(x)/white^2) / (1 + L(x))
            ldr_image[:, :, i] = scaled[:, :, i] * (1 + scaled[:, :, i] / (white_levels[i] * white_levels[i])) / (1 + scaled[:, :, i])
        
        # Clip values to [0, 1] range
        ldr_image = np.clip(ldr_image, 0, 1)
        
        return ldr_image
    
    def tone_map_luminance(self, hdr_image):
        """
        Apply tone mapping to luminance channel only, preserving chromaticity.
        
        Args:
            hdr_image (numpy.ndarray): Input HDR image
            
        Returns:
            numpy.ndarray: Tone mapped LDR image
        """
        # Extract luminance
        luminance = self.rgb_to_xyz(hdr_image)
        
        # Calculate log average luminance
        l_avg = self.log_average_luminance(hdr_image, use_rgb=False)
        
        # Scale luminance by key value / log average
        scaled_luminance = (self.key / l_avg) * luminance
        
        # Calculate white level
        white_level = self.burn * np.max(luminance)
        
        # Apply tone mapping operator to luminance
        mapped_luminance = scaled_luminance * (1 + scaled_luminance / (white_level * white_level)) / (1 + scaled_luminance)
        
        # Preserve chromaticity (color ratios)
        ldr_image = np.zeros_like(hdr_image)
        
        # For pixels with non-zero luminance, scale RGB values by the ratio of mapped to original luminance
        valid_pixels = luminance > 0
        for i in range(3):
            ldr_image[:, :, i] = np.zeros_like(luminance)
            ldr_image[:, :, i][valid_pixels] = hdr_image[:, :, i][valid_pixels] * (mapped_luminance[valid_pixels] / luminance[valid_pixels])
        
        # Clip values to [0, 1] range
        ldr_image = np.clip(ldr_image, 0, 1)
        
        return ldr_image
    
    def apply_gamma(self, image, gamma=2.2):
        """
        Apply gamma correction to the tone mapped image.
        
        Args:
            image (numpy.ndarray): Tone mapped image
            gamma (float): Gamma value (2.2 is standard)
            
        Returns:
            numpy.ndarray: Gamma-corrected image
        """
        return np.power(image, 1.0 / gamma)
    
    def tone_map(self, hdr_image, method='luminance', apply_gamma_correction=True):
        """
        Apply tone mapping to an HDR image.
        
        Args:
            hdr_image (numpy.ndarray): Input HDR image
            method (str): Tone mapping method, either 'rgb' or 'luminance'
            apply_gamma_correction (bool): Whether to apply gamma correction
            
        Returns:
            numpy.ndarray: Tone mapped LDR image
        """
        # Check that image is in the correct format
        if hdr_image.dtype != np.float32 and hdr_image.dtype != np.float64:
            hdr_image = hdr_image.astype(np.float32)
            
        # Ensure there are no negative values
        hdr_image = np.maximum(hdr_image, 0)
        
        # Apply tone mapping
        if method == 'rgb':
            ldr_image = self.tone_map_rgb(hdr_image)
        elif method == 'luminance':
            ldr_image = self.tone_map_luminance(hdr_image)
        else:
            raise ValueError("Method must be either 'rgb' or 'luminance'")
        
        # Apply gamma correction if requested
        if apply_gamma_correction:
            ldr_image = self.apply_gamma(ldr_image)
        
        return ldr_image
    
    def visualize_tone_mapping(self, hdr_image, gamma_values=[1.0, 2.2], key_values=[0.09, 0.18, 0.36], method='luminance'):
        """
        Visualize tone mapping with different parameters.
        
        Args:
            hdr_image (numpy.ndarray): Input HDR image
            gamma_values (list): List of gamma values to try
            key_values (list): List of key values to try
            method (str): Tone mapping method, either 'rgb' or 'luminance'
        """
        n_gamma = len(gamma_values)
        n_key = len(key_values)
        
        fig, axes = plt.subplots(n_key, n_gamma, figsize=(n_gamma * 4, n_key * 4))
        
        for i, key in enumerate(key_values):
            for j, gamma in enumerate(gamma_values):
                # Update key value
                self.key = key
                
                # Apply tone mapping
                ldr_image = self.tone_map(hdr_image, method=method, apply_gamma_correction=False)
                
                # Apply gamma
                gamma_corrected = self.apply_gamma(ldr_image, gamma)
                
                # Display
                if n_key == 1 and n_gamma == 1:
                    ax = axes
                elif n_key == 1:
                    ax = axes[j]
                elif n_gamma == 1:
                    ax = axes[i]
                else:
                    ax = axes[i, j]
                
                ax.imshow(gamma_corrected)
                ax.set_title(f'Key: {key}, Gamma: {gamma}')
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    def save_tone_mapped_image(self, ldr_image, filename):
        """
        Save tone mapped image to file.
        
        Args:
            ldr_image (numpy.ndarray): Tone mapped image
            filename (str): Output filename
        """
        # Convert to uint8
        ldr_image_8bit = (ldr_image * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        if ldr_image.shape[2] == 3:
            ldr_image_8bit = cv2.cvtColor(ldr_image_8bit, cv2.COLOR_RGB2BGR)
        
        # Save image
        cv2.imwrite(filename, ldr_image_8bit) 