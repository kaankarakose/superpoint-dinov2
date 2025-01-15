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
OR, use `builder.sh` to build the image
```bash
bash builder.sh
```
3. Launch container
```bash
 bash launcher.sh
```
4. In the container, run:
```bash
conda init
source /opt/conda/etc/profile.d/conda.sh
conda activate dinov2
```
5. Clone the repo:
```
git clone git@github.com:kaankarakose/superpoint-dinov2.git
cd superpoint-dinov2
```

## Usage

The main workflow is demonstrated in `process_image.py`. Here's a basic example of how to use the feature matcher:

```python
# Example code will be added based on scripts/process_image.py implementation
superpoint_extractor = SuperPointFeatureExtractor()
dino_extractor = DistributedDinov2FeatureExtractor(model_name='dinov2_vits14')
processor = ImageFeatureProcessor(dino_extractor, superpoint_extractor)
image1_path = './test_images/image0.jpg'
image2_path = './test_images/image1.jpg'
#reading image as numpy array
img1 = cv.imread(image1_path)          
img2 = cv.imread(image2_path)  

# get keypoints and its corresponded descriptors from Dinov2
img1_feature = processor.process_single_image(image1_path)
img2_feature = processor.process_single_image(image2_path)

# get similarities
matches, distances = improved_feature_matching(img1_feature['descriptors'], img2_feature['descriptors'], threshold=0.6)
# use RANSAC to discard outliers
filtered_matches = filter_matches_with_ransac(img1_feature['keypoints'], img2_feature['keypoints'], matches, threshold=5.0)
#plot final result
plot_keypoint_matches(img1, img2, img1_feature['keypoints'], img2_feature['keypoints'], filtered_matches, output_path='./output/matches.png')  
```

### Key Components

- **SuperPoint**: Used for keypoint detection
- **DINOv2**: Provides robust visual descriptors
- **FAISS**: Efficient similarity search for feature matching
- **RANSAC**: Outlier removal for robust matching

## Features

- Combines keypoint detection with self-supervised visual features
- Fast similarity search using FAISS
- Outlier removal with RANSAC
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

Feel free to open issues or submit pull requests. Contributions are welcomed!

## Citation

If you use this code in your research, please cite:
```
@misc{superpoint-dinov2,
    author = {Kaan Karakose},
    title = {SuperPoint Keypoints with DINOv2 Descriptors},
    year = {2025},
    publisher = {GitHub},
    howpublished = {https://github.com/kaankarakose/superpoint-dinov2}
}
```
## License

This project is licensed under the MIT License.

## Acknowledgments
Many thanks to these great works!
- [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork)
- [DINOv2](https://github.com/facebookresearch/dinov2) 
- [FAISS](https://github.com/facebookresearch/faiss)
  
