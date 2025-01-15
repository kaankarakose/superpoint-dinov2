import torch
from kornia.feature import DeDoDe
from PIL import Image
import numpy as np

class DeDoDeFeatureExtractor:
    def __init__(self, device="cuda", detector_weights="L-C4-v2", descriptor_weights="B-upright"):
        """
        Initialize DeDoDe feature extractor
        Args:
            device (str): Device to run the model on ('cuda' or 'cpu')
            detector_weights (str): Type of detector weights ('
            ' recommended)
            descriptor_weights (str): Type of descriptor weights ('B-upright' or 'G-upright')
        """
        self.device = device
        self.model = DeDoDe.from_pretrained(
            detector_weights=detector_weights,
            descriptor_weights=descriptor_weights
        ).to(device)
        self.model.eval()
        
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                std=(0.229, 0.224, 0.225)),
            ])

    def extract_keypoints(self, image):
        """
        Extract keypoints from a single image.
        Args:
            image (PIL.Image): Input image
        Returns:
            tuple: (keypoints, scores, descriptors)
        """
    # Convert PIL image to torch tensor
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features using DeDoDe's combined interface
        with torch.no_grad():
            keypoints, scores, descriptors = self.model(img_tensor)
        
        # Convert to numpy and move to CPU
        keypoints = keypoints[0].cpu().numpy()  # Shape: (N, 2)
        scores = scores[0].cpu().numpy()        # Shape: (N,)
        descriptors = descriptors[0].cpu().numpy()  # Shape: (N, D)

        # Scale keypoints to image dimensions since DeDoDe returns normalized coordinates
        h, w = image.size[1], image.size[0]
        keypoints[:, 0] *= w  # Scale x coordinates
        keypoints[:, 1] *= h  # Scale y coordinates

        return keypoints, scores, descriptors
    
    def extract_keypoints_from_path(self, image_path):
        """
        Extract keypoints from an image file.
        Args:
            image_path (str): Path to the image file
        Returns:
            tuple: (keypoints, scores, descriptors)
        """
        image = Image.open(image_path)
        return self.extract_keypoints(image)