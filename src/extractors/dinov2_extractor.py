import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2

## DEBUG
import inspect

class DistributedDinov2FeatureExtractor:
    def __init__(self, 
                 repo_name="facebookresearch/dinov2",
                 model_name="dinov2_vitg14",
                 resize = None,
                 register_token_weight = 0.3):
        
        self.repo_name = repo_name
        self.model_name = model_name
        self.resize = resize
        ## Register Weight
        self.register_token_weight = register_token_weight
        # Setup multi-GPU without process group
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name)
        
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        
        self.model.to(self.device)
        self.model.eval()
        
        if self.resize:
            self.transform = transforms.Compose([
                transforms.Resize(self.resize),  # Standard ImageNet size
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                std=(0.229, 0.224, 0.225)),
            ])
        else:
             self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                              std=(0.229, 0.224, 0.225)),
        ])

    def prepare_image(self, rgb_image_numpy):
        image = Image.fromarray(rgb_image_numpy)
        image_tensor = self.transform(image)
        
        # Get the actual model (removing DataParallel wrapper if present)
        actual_model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        
        resize_scale_w = image.width / image_tensor.shape[2]
        resize_scale_h = image.height / image_tensor.shape[1]
        
        height, width = image_tensor.shape[1:]
        cropped_width = width - width % actual_model.patch_size # actual_model.patch_size is 14
        cropped_height = height - height % actual_model.patch_size
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]
        
        grid_size = (cropped_height // actual_model.patch_size, 
                    cropped_width // actual_model.patch_size)
        resize_scales = (resize_scale_w, resize_scale_h)
        
        return image_tensor, grid_size, resize_scales

    # def extract_features(self, image_tensor, n = 1):
    #     image_batch = image_tensor.unsqueeze(0).to(self.device)
        
    #     with torch.no_grad():
    #         actual_model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
            
            
    #         # Print shape before intermediate_features
    #         print("Image batch shape:", image_batch.shape)
    #         # Get intermediate features 
    #         intermediate_features = actual_model.get_intermediate_layers(image_batch, n = n) #n is last how many layers I want? n x Pathes X Embeded size
    #         # Print shape after get_intermediate_layers
    #         print("Intermediate features shape:", [f.shape for f in intermediate_features])
    #         # Get register tokens from forward_features
    #         if self.model_name.endswith('_reg') or self.model_name + "_reg" == self.model_name:
    #             features_dict = actual_model.forward_features(image_batch)
                
    #             # Extract normalized tokens - only patch tokens, skip cls token
    #             patch_tokens = features_dict['x_norm_patchtokens']
    #             features = patch_tokens.squeeze(0)  # Remove batch dimension
    #         else:
    #             # For non-register model, remove cls token (first token)
    #             features = intermediate_features#  simdide[1:] olmasi daha dogru !!![1:]!!!  # Skip the first (CLS) token
            
    #     return features[0].squeeze(0).cpu().detach().numpy()
    
    
    def extract_features(self, image_tensor, n = 1, return_class_token= False,norm = True):
        image_batch = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            actual_model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
            # Print shape before intermediate_features
            #print("Image batch shape:", image_batch.shape)
            # Get intermediate features 
            intermediate_features = actual_model.get_intermediate_layers(image_batch,norm=norm,return_class_token=return_class_token,n=n)# n is last how many layers I want? n x Pathes X Embeded size
            # else:
            #     print("Intermediate features shape:", [f.shape for f in intermediate_features])
            # Get register tokens from forward_features
            if self.model_name.endswith('_reg') or self.model_name + "_reg" == self.model_name:
                features_dict = actual_model.forward_features(image_batch)
                
                # Extract normalized tokens - only patch tokens, skip cls token
                patch_tokens = features_dict['x_norm_patchtokens']
                features = patch_tokens.squeeze(0)  # Remove batch dimension
            else:
                # For non-register model, remove cls token (first token)
                features = intermediate_features # [1:]  # Skip the first (CLS) token!!!
            
        #return features[0].squeeze(0).cpu().detach().numpy()
        if isinstance(features, torch.Tensor):
            return features.detach().numpy()#[0].squeeze(0).cpu().detach().numpy()
        else:
            stacked_features = torch.stack(features)

            return stacked_features.cpu().squeeze(1).detach().numpy()

    def get_embedding_visualization(self, tokens, grid_size , resized_mask = None):
        pca = PCA(n_components = 3)
        if resized_mask is not None:
            tokens = tokens[resized_mask]
        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
        if resized_mask is not None:
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        reduced_tokens = reduced_tokens.reshape((*grid_size, -1))
        normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))
        return normalized_tokens