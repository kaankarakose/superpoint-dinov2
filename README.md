# SuperPoint-DINOv2 Feature Matcher

This repository combines SuperPoint keypoint detection with DINOv2 descriptors for robust feature matching. The project leverages the strengths of both approaches: SuperPoint's accurate keypoint detection and DINOv2's powerful self-supervised visual features.

## Overview

The project works by:
1. Detecting keypoints using SuperPoint
2. Extracting DINOv2 patch descriptors
3. Aligning DINOv2 patches with SuperPoint keypoints
4. Performing feature matching using FAISS
5. Filtering matches with RANSAC to remove outliers

## Installation

This project uses Apptainer (formerly Singularity) for containerization to ensure reproducibility and easy setup. To install:

1. First, ensure you have Apptainer installed on your system
2. Build the container using the provided definition file:
```bash
apptainer build superpoint_dinov2.sif superpoint_dinov2.def
```

## Usage

The main workflow is demonstrated in `process_image.py`. Here's a basic example of how to use the feature matcher:

```python
# Example code will be added based on process_image.py implementation
```

### Key Components

- **SuperPoint**: Used for keypoint detection
- **DINOv2**: Provides robust visual descriptors
- **FAISS**: Efficient similarity search for feature matching
- **RANSAC**: Outlier removal for robust matching

## Features

- Combines state-of-the-art keypoint detection with self-supervised visual features
- Fast similarity search using FAISS
- Robust outlier removal with RANSAC
- Containerized environment for easy deployment

## Results

Below is an example of feature matching results using our SuperPoint-DINOv2 combination:

![Feature Matching Result](output/matches.png)

The image above demonstrates the matching between two images using SuperPoint keypoints and DINOv2 descriptors, with RANSAC filtering to remove outliers.

## TODO
- [ ] Clear the unused functions
- [ ] Add Gradio demo interface for quick testing and visualization
  - Upload two images
  - Display matched keypoints
  - Show matching results
  - Interactive parameter adjustment

## Contributing

Feel free to open issues or submit pull requests. We welcome contributions to improve the project!

## License

This project is licensed under the MIT License.


## Acknowledgments

- [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork) implementation
- [DINOv2](https://github.com/facebookresearch/dinov2) team
- [FAISS](https://github.com/facebookresearch/faiss) library