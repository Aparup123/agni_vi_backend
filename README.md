# PSR-crater-detection# Enhancement of Permanently Shadowed Regions (PSR) of Lunar Craters Captured by OHRC of Chandrayaan-2

## Project Overview
This project focuses on enhancing low-light images of Permanently Shadowed Regions (PSR) on the Moon's surface captured by Chandrayaan-2's OHRC (Orbiter High-Resolution Camera). The goal is to improve the signal-to-noise ratio (SNR) of these dimly lit areas to aid in lunar surface analysis, landing site selection, and geomorphological research.

### Key Features:
- **Light Enhancement**: Techniques like CLAHE, Histogram Equalization, and Retinex-based methods.
- **Noise Reduction**: Using OpenCV's noise reduction filters like Non-local Means, Gaussian Blurring, etc.
- **Edge Detection**: Canny and Sobel operators are used to detect the edges of lunar craters and features.
- **Feature Detection & Labeling**: Automatic and manual detection and labeling of lunar surface features.
- **Web Interface Mockup**: A simple web UI mockup for visualization of the results.
  

## Getting Started

### Prerequisites
- Python 3.x
- OpenCV
- Matplotlib, Numpy, SciPy (for processing and plotting)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/PSR-enhancement.git
   cd PSR-enhancement

