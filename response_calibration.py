import numpy as np
from weighting_schemes import WeightingSchemes

class ResponseCalibration:
    def __init__(self, images, exposure_times, weighting_method='gaussian', z_min=0.05, z_max=0.95, l=50):
        """
        Initialize the response calibration solver.
        
        Args:
            images (list): List of LDR images (numpy arrays)
            exposure_times (list): List of exposure times for each image
            weighting_method (str): Weighting method to use ('uniform', 'tent', 'gaussian', 'photon')
            z_min (float): Minimum intensity value (default: 0.05)
            z_max (float): Maximum intensity value (default: 0.95)
            l (float): Lambda smoothing term weight (default: 50)
        """
        self.images = images
        self.exposure_times = np.array(exposure_times)
        self.l = l
        self.weights = WeightingSchemes(z_min, z_max)
        self.weighting_method = getattr(self.weights, weighting_method)
        self.z_min = z_min
        self.z_max = z_max
        
    def sample_pixels(self, n_samples=200):
        """
        Sample pixels from images for calibration.
        
        Args:
            n_samples (int): Number of pixels to sample per image
            
        Returns:
            tuple: (sampled pixel values, corresponding exposure times)
        """
        # Get image dimensions
        h, w = self.images[0].shape[:2]
        
        # Sample random pixel locations
        i_samples = np.random.randint(0, h, n_samples)
        j_samples = np.random.randint(0, w, n_samples)
        
        # Extract pixel values from all images
        z_samples = []
        exposure_times_samples = []
        
        for img, t in zip(self.images, self.exposure_times):
            z = img[i_samples, j_samples]
            z_samples.extend(z.flatten())
            exposure_times_samples.extend([t] * len(z.flatten()))
            
        return np.array(z_samples), np.array(exposure_times_samples)
    
    def solve_g(self, n_samples=200):
        """
        Solve for the camera response function g.
        
        Args:
            n_samples (int): Number of pixels to sample per image
            
        Returns:
            numpy.ndarray: The recovered response function g
        """
        # Sample pixels
        z_values, t_values = self.sample_pixels(n_samples)
        
        # Number of unique pixel values (typically 256 for 8-bit images)
        n = 256
        
        # Create the coefficient matrix A and the vector b
        n_data = len(z_values)
        n_unknowns = n + n_samples  # g values + log irradiances
        
        # Initialize A matrix and b vector
        A = np.zeros((n_data + n-2, n_unknowns))
        b = np.zeros(n_data + n-2)
        
        # Fill in data term equations
        k = 0
        for i, (z, t) in enumerate(zip(z_values, t_values)):
            # Convert z from [0,1] to [0,255] range for indexing
            z_idx = int(z * 255)
            w = self.weighting_method(z)  # z is already in [0,1] for weighting
            A[k, z_idx] = w
            A[k, n + i//len(self.images)] = -w
            b[k] = w * np.log(t)
            k += 1
            
        # Add smoothness terms
        for i in range(n-2):
            A[k, i:i+3] = self.l * np.array([1, -2, 1])
            k += 1
            
        # Add constraint g[127] = 0 (middle value)
        A[k-1, 127] = 1
        
        # Solve the system using least squares
        solution = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # Extract g function (first n values)
        g = solution[:n]
        
        return g
        
    def linearize_image(self, image, g):
        """
        Convert a non-linear image to linear using the recovered response function g.
        
        Args:
            image (numpy.ndarray): Input non-linear image
            g (numpy.ndarray): Recovered response function
            
        Returns:
            numpy.ndarray: Linearized image
        """
        return np.exp(g[image]) 