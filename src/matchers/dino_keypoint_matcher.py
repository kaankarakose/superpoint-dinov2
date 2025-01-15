from PIL import Image
import numpy as np
import os
class DINOSuperPointMatcher:
    def __init__(self, dino_extractor, superpoint_extractor):
        """
        Initialize the matcher with both feature extractors.
        Args:
            dino_extractor (Dinov2FeatureExtractor): Instance of DINO feature extractor
            superpoint_extractor (SuperPointFeatureExtractor): Instance of SuperPoint feature extractor
        """
        self.dino = dino_extractor # Dinov2FeatureExtractor
        self.superpoint = superpoint_extractor #SuperPointFeatureExtractor

    def _tensor_to_numpy(self,image_tensor):
        # 1. Clone and detach if tensor is in GPU
        img = image_tensor.cpu().detach().clone()
    
        # 2. Denormalize 
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
        img = img * std + mean
        
        # 3. Clamp values to [0, 1]
        img = torch.clamp(img, 0, 1)
        
        # 4. Convert to numpy and transpose from [C,H,W] to [H,W,C]
        img = img.numpy()
        img = np.transpose(img, (1, 2, 0))
        
        # 5. Scale to [0, 255] and convert to uint8
        img = (img * 255).astype(np.uint8)
    
        return img
    def match_keypoints_to_patches(self, image_numpy):
        """
        Match SuperPoint keypoints to DINO patches and replace descriptors.
        Args:
            image_numpy (np.ndarray): Input image in numpy format (H,W,C)
        Returns:
            tuple: (keypoints, scores, dino_descriptors, patch_indices)
        """
        ## Convert numpy image to PIL for SuperPoint
         # Store original dimensions
        original_h, original_w = image_numpy.shape[:2] # If image [5180 X 3444]
        
        # Get DINO features and grid information
        # Process image for DINO
        dino_tensor, grid_size, dino_scales = self.dino.prepare_image(image_numpy)
        #dino_tensor [5180 X 3444], [5180/14 X 3444/14]
        dino_h, dino_w = dino_tensor.shape[1:3]  # Get DINO processed image dimensions
        
    
        # Extract DINO features

        dino_features = self.dino.extract_features(dino_tensor,
                                                   n = 1, # Number of outputs from end of the 
                                                    return_class_token= False,
                                                    norm = True)#[1:]  # Skip CLS token for intermadiate forward !! DINO
        
        # Process original image with SuperPoint
        # DinoFeatures = > (5180/14) * (3444/14)
        dino_features = dino_features.squeeze(0) 
    
    
        image_pil = Image.fromarray(image_numpy)
        keypoints, scores, _ = self.superpoint.extract_keypoints(image_pil)
        # Reshape scores to (N, 1)
        scores = scores.reshape(-1, 1)
    


        # Calculate cell size for 14x14 grid in original image dimensions
        cell_height = original_h / grid_size[0]  # height of each grid cell
        cell_width = original_w / grid_size[1]   # width of each grid cell

        # Add assertions to verify cell sizes
        assert int(cell_height) == 14, f"Cell height should be 14, but got {cell_height}"
        assert int(cell_width) == 14, f"Cell width should be 14, but got {cell_width}"
                
        # Match keypoints to patches
        patch_indices = []
        dino_descriptors = []
        
        for keypoint in keypoints:
            x, y = keypoint
            
            # Find which grid cell this keypoint falls into
            grid_x = int(x // cell_width)   
            grid_y = int(y  // cell_height)  
            
            # Calculate linear index
            patch_idx = grid_y * grid_size[1] + grid_x
            
            patch_indices.append(patch_idx)
            dino_descriptors.append(dino_features[patch_idx])
        assert len(keypoints) == len(dino_descriptors), f"Should be same lenght"
        
        return keypoints, scores.reshape(-1, 1), np.array(dino_descriptors), np.array(patch_indices), dino_scales, grid_size



    def idx_to_grid_position(self,patch_idx, original_image_height, patch_size=14):
        """
        Convert patch index to grid position where grid_width is determined by image dimensions.
        
        Args:
            patch_idx (int or np.ndarray): Patch index(es) to convert
            original_image_height (int): Height of the original image
            patch_size (int): Size of each patch in pixels, default 14
        
        Returns:
            tuple: (row, col) representing position in the grid
        """
    
        # Calculate grid width based on original image height
        grid_width = original_image_height // patch_size
        
        # Convert to numpy array if not already
        patch_idx = np.array(patch_idx)
        
        # Calculate row and column
        row = patch_idx // grid_width
        col = patch_idx % grid_width
        
        return row, col
   

    def visualize_matches(self, image_numpy, keypoints, patch_indices, grid_size):
        """
        Visualize keypoints and their corresponding DINO patches.
        Args:
            image_numpy (np.ndarray): Input image
            keypoints (np.ndarray): Array of keypoint coordinates
            patch_indices (np.ndarray): Array of corresponding patch indices
            grid_size (tuple): Size of the DINO grid (H,W)
        Returns:
            np.ndarray: Visualization image with keypoints and patch boundaries
        """
        
        vis_image = image_numpy.copy()
        
        # Draw grid
        h, w = image_numpy.shape[:2]
        cell_h, cell_w = h / grid_size[0], w / grid_size[1]
        # Draw vertical lines
        for i in range(int(cell_w)): ## 140 / 14 = 10
            x = int(i * grid_size[0])
            cv2.line(vis_image, (x, 0), (x, h), (0, 255, 0), 1)
    
        # Draw horizontal lines
        for i in range(int(cell_h)):
            y = int(i * grid_size[0])
            cv2.line(vis_image, (0, y), (w, y), (0, 255, 0), 1)
            
        # Draw keypoints
        for keypoint in keypoints:
            x, y = int(keypoint[0]), int(keypoint[1])
            cv2.circle(vis_image, (x, y), 8, (0, 0, 255), -1)
            
        return vis_image