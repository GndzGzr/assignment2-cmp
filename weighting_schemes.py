import numpy as np

class WeightingSchemes:
    def __init__(self, z_min=0.05, z_max=0.95):
        """
        Initialize weighting schemes with clipping values.
        
        Args:
            z_min (float): Minimum intensity value (default: 0.05)
            z_max (float): Maximum intensity value (default: 0.95)
        """
        self.z_min = z_min
        self.z_max = z_max
    
    def _clip_values(self, z):
        """Helper function to check if values are within valid range."""
        return np.logical_and(z >= self.z_min, z <= self.z_max)
    
    def uniform(self, z):
        """
        Uniform weighting scheme.
        
        Args:
            z (numpy.ndarray): Input intensity values in range [0,1]
            
        Returns:
            numpy.ndarray: Weights (1 for valid range, 0 otherwise)
        """
        return self._clip_values(z).astype(float)
    
    def tent(self, z):
        """
        Tent weighting scheme.
        
        Args:
            z (numpy.ndarray): Input intensity values in range [0,1]
            
        Returns:
            numpy.ndarray: Weights based on tent function
        """
        weights = np.minimum(z, 1 - z)
        return np.where(self._clip_values(z), weights, 0)
    
    def gaussian(self, z):
        """
        Gaussian weighting scheme.
        
        Args:
            z (numpy.ndarray): Input intensity values in range [0,1]
            
        Returns:
            numpy.ndarray: Weights based on Gaussian function
        """
        weights = np.exp(-4 * ((z - 0.5) ** 2) / (0.5 ** 2))
        return np.where(self._clip_values(z), weights, 0)
    
    def photon(self, z, k=1):
        """
        Photon weighting scheme.
        
        Args:
            z (numpy.ndarray): Input intensity values in range [0,1]
            k (float): Exposure time exponent (default: 1)
            
        Returns:
            numpy.ndarray: Weights based on photon noise model
        """
        weights = z ** k
        return np.where(self._clip_values(z), weights, 0) 