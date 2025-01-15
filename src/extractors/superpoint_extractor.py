
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
import torch
import numpy as np
class SuperPointFeatureExtractor:
    def __init__(self, device="cuda"):
        """
        Initialize SuperPoint feature extractor
        Args:
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
        self.model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint").to(self.device)
        self.model.eval()
        
        
    def extract_keypoints(self, image):
        """
        Extract keypoints from a single image.
        Args:
            image (PIL.Image): Input image
        Returns:
            tuple: (keypoints, scores, descriptors)
        """
        # Get original image dimensions
        orig_h, orig_w = image.size[1], image.size[0]
        
        # Get target size from processor config
        target_h = self.processor.size['height']
        target_w = self.processor.size['width']
        
        # Calculate scaling factors
        scale_h = orig_h / target_h
        scale_w = orig_w / target_w
        
        # Process image
        inputs = self.processor(image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get valid keypoints using the mask
        mask = outputs.mask[0]
        indices = torch.nonzero(mask).squeeze()
        
        # Get keypoints and scale them back to original image dimensions
        keypoints = outputs.keypoints[0][indices].cpu().numpy()
        keypoints[:, 0] *= scale_w * target_w # Scale x coordinates
        keypoints[:, 1] *= scale_h * target_h # Scale y coordinates 

        scores = outputs.scores[0][indices].cpu().numpy()
        descriptors = outputs.descriptors[0][indices].cpu().numpy()
        
        return keypoints, scores, descriptors