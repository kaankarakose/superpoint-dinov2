SuperPoint-DINOv2 Feature Matcher
This repository combines SuperPoint keypoint detection with DINOv2 descriptors for feature matching. The project leverages the strengths of both approaches: SuperPoint's accurate keypoint detection and DINOv2's powerful self-supervised visual features.
Overview
The project works by:

Detecting keypoints using SuperPoint
Extracting DINOv2 patch descriptors
Aligning DINOv2 patches with SuperPoint keypoints
Performing feature matching using FAISS
Filtering matches with RANSAC to remove outliers

Installation
This project uses Apptainer (formerly Singularity) for containerization to ensure reproducibility and easy setup. To install:

First, ensure you have Apptainer installed on your system
Build the container using the provided definition file:

bashCopyapptainer build superpoint_dinov2.sif superpoint_dinov2.def
Usage
The main workflow is demonstrated in process_image.py. Here's a basic example of how to use the feature matcher:
pythonCopy# Example code will be added based on process_image.py implementation
Key Components

SuperPoint: Used for keypoint detection
DINOv2: Provides robust visual descriptors
FAISS: Efficient similarity search for feature matching
RANSAC: Outlier removal for robust matching

Features

Combines state-of-the-art keypoint detection with self-supervised visual features
Fast similarity search using FAISS
Robust outlier removal with RANSAC
Containerized environment for easy deployment

Contributing
Feel free to open issues or submit pull requests. We welcome contributions to improve the project!
License
[License information to be added]
Citation
If you use this code in your research, please cite:
Copy[Citation information to be added]
Acknowledgments

SuperPoint implementation
DINOv2 team
FAISS library