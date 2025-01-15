from tqdm import tqdm
import os
from src.matchers.dino_keypoint_matcher import DINOSuperPointMatcher
import numpy as np
import json
import cv2

class ImageFeatureProcessor:
    def __init__(self, dino_extractor, superpoint_extractor, device="cuda"):
        self.matcher = DINOSuperPointMatcher(dino_extractor, superpoint_extractor)
        self.device = device
        self._cached_features = {}

    def load_and_process_directory(self, base_folder, save_dir=None):
            image_data = {}
            
            for object_name in os.listdir(base_folder):
                object_folder = os.path.join(base_folder, object_name)
                if not os.path.isdir(object_folder):
                    continue
                    
                image_data[object_name] = {}
                for ht in ['H1', 'H2', 'H3', 'H4']:
                    current_folder = os.path.join(object_folder, ht)
                    if not os.path.exists(current_folder):
                        continue
                        
                    image_paths = [
                        os.path.join(current_folder, img) 
                        for img in os.listdir(current_folder) 
                        if img.endswith('.JPG')
                    ]
                    
                    batch_data = self._process_image_batch(image_paths)
                    image_data[object_name][ht] = batch_data
                    
                    if save_dir:
                        self._save_batch_data(batch_data, object_name, ht, save_dir)
            
            return image_data
        
        
    def load_and_process_directory(self, base_folder, save_dir=None):
            image_data = {}
            
            for object_name in os.listdir(base_folder):
                object_folder = os.path.join(base_folder, object_name)
                if not os.path.isdir(object_folder):
                    continue
                    
                image_data[object_name] = {}
                for ht in ['H1', 'H2', 'H3', 'H4']:
                    current_folder = os.path.join(object_folder, ht)
                    if os.path.exists(os.path.join(save_dir, object_name, ht)):
                        print(f"{save_dir}/{object_folder}/{ht} is exist. Skipping.")
                        continue # 
                    
                    if not os.path.exists(current_folder):
                        continue
                        
                    image_paths = [
                        os.path.join(current_folder, img) 
                        for img in os.listdir(current_folder) 
                        if img.endswith('.JPG')
                    ]
                    
                    batch_data = self._process_image_batch(image_paths)
                    image_data[object_name][ht] = batch_data
                    
                    if save_dir:
                        self._save_batch_data(batch_data, object_name, ht, save_dir)
            
            return image_data

    def _save_batch_data(self, batch_data, object_name, ht, save_dir):
        
        object_save_dir = os.path.join(save_dir, object_name, ht)
        features_dir = os.path.join(object_save_dir, 'features')
        os.makedirs(features_dir, exist_ok=True)

        metadata = []
        for data in tqdm(batch_data):
            base_name = os.path.splitext(os.path.basename(data['path']))[0]
            
            # Save features separately
            feature_file = os.path.join(features_dir, f"{base_name}_features.npz")
            np.savez_compressed(feature_file, 
                keypoints=data['keypoints'],
                descriptors=data['descriptors'],
                scores=data['scores'],
                indices=data['indices']
            )
            
            # Save only JSON-serializable data in metadata
            metadata.append({
                'image_path': data['path'],
                'feature_file': os.path.join('features', f"{base_name}_features.npz"),
                'grid_size': data['grid_size'].tolist() if isinstance(data['grid_size'], np.ndarray) else data['grid_size'],
                'scales': data['scales'].tolist() if isinstance(data['scales'], np.ndarray) else data['scales'],
                'num_keypoints': len(data['keypoints']),
                'descriptor_dim': data['descriptors'].shape[1]
            })
        
        with open(os.path.join(object_save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _process_image_batch(self, image_paths):
        """Process a batch of images and extract features."""
        batch_data = []
        
        for path in tqdm(image_paths):
            print('processing')
            if path in self._cached_features:
                batch_data.append(self._cached_features[path])
                continue
                
            # Load and process image
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract features
            keypoints, scores, descriptors, indices, scales, grid_size = (
                self.matcher.match_keypoints_to_patches(image)
            )
            
            data = {
                'path': path,
                'keypoints': keypoints,
                'descriptors': descriptors,
                'scores': scores,
                'indices': indices,
                'scales': scales,
                'grid_size': grid_size
            }
            
            self._cached_features[path] = data
            batch_data.append(data)
            
        return batch_data

    def process_single_image(self, image_path, output_dir=None):
        """Process a single image and optionally save results."""
        if image_path in self._cached_features:
            return self._cached_features[image_path]
            
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        keypoints, scores, descriptors, indices, dino_scales, grid_size = self.matcher.match_keypoints_to_patches(image) # RETURNS  keypoints, scores.reshape(-1, 1), np.array(dino_descriptors), np.array(patch_indices), dino_scales,grid_size
        
        # if output_dir:
        #     os.makedirs(output_dir, exist_ok=True)
        #     base_name = os.path.splitext(os.path.basename(image_path))[0]
        #     np.save(os.path.join(output_dir, f"{base_name}_features.npy"), {
        #         'keypoints': results[0],
        #         'descriptors': results[2],
        #         'indices': results[3]

        return {'keypoints': keypoints, 'descriptors': descriptors, 'indices': indices}