# HDR Image Generation Experiments

This project implements various methods for HDR (High Dynamic Range) image generation from multiple exposures. It includes:

1. Weighting schemes (uniform, tent, Gaussian, photon)
2. Camera response calibration
3. Linear and logarithmic merging methods
4. Comprehensive experiments with different combinations

## Running the Experiments

To run all experiments, simply execute:

```bash
python hdr_experiments.py
```

This will:
1. Load both TIFF (RAW) and JPG (rendered) images from the `data/Lab Booth` directory
2. Downscale them for faster processing (10% of original size)
3. For JPG images: calibrate a camera response function and linearize them
4. Generate 16 different HDR images (2 image types × 2 merging methods × 4 weighting schemes)
5. Save all results in the `results` directory

## Interpreting Results

After running the experiments, you'll find these results in the `results` directory:

1. **Individual HDR Images**: Named according to their parameters (e.g., `jpg_linear_gaussian.png`)
2. **Comparison Grid**: A visual grid showing all 16 HDR images (`all_hdr_comparisons.png`)
3. **Timing Table**: A table showing processing times for all methods (`timing_table.png`)
4. **Response Function**: If JPG images were processed, a plot of the recovered camera response function

## Components

The experiment uses several Python modules:

- `weighting_schemes.py`: Implements four different pixel weighting schemes
- `response_calibration.py`: Implements the Debevec & Malik method for camera response recovery
- `hdr_merging.py`: Implements linear and logarithmic HDR merging methods
- `hdr_experiments.py`: Orchestrates experiments across different combinations

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- OpenCV

## Notes

- The HDR images saved are tone-mapped for display purposes only. The actual HDR data has much higher dynamic range.
- For the original full-resolution processing, increase the `scale_factor` parameter in the `downscale_images` function.
- The exposure times are simulated (doubling for each image) - in a real workflow, extract these from EXIF data. 