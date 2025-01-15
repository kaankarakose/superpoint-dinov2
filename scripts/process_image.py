
import os
import cv2
from src.extractors.superpoint_extractor import SuperPointFeatureExtractor
from src.extractors.dinov2_extractor import DistributedDinov2FeatureExtractor


from src.processors.image_processor import ImageFeatureProcessor

from PIL import Image
from tqdm import tqdm

import faiss
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from src.vis.utils import plot_keypoint_matches




def improved_feature_matching(db_features, query_features, threshold=0.8):
    
    nlist = max(1, len(db_features) // 2**5) ## experimental
    # print("nlist", nlist)
    # print("DB features shape:", db_features.shape)
    # print("Query features shape:", query_features.shape)
    # print("Are there any NaN values in DB?", np.isnan(db_features).any())
    # print("Are there any NaN values in Query?", np.isnan(query_features).any())
    # Normalize features
    db_features = db_features.astype('float32')
    query_features = query_features.astype('float32')
    db_features = db_features / np.linalg.norm(db_features, axis=1)[:, np.newaxis]
    query_features = query_features / np.linalg.norm(query_features, axis=1)[:, np.newaxis]

    # Build FAISS index with IVF (Inverted File Index) for faster search
    d = db_features.shape[1]  # dimension
    quantizer = faiss.IndexFlatL2(d)
    #index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = min(nlist, 8) 
    
    try:
        # Train the index
        index.train(db_features)
        if not index.is_trained:
            raise RuntimeError("Index training failed")
            
        # Add vectors to the index
        index.add(db_features)
        
        # Verify vectors were added
        if index.ntotal == 0:
            raise RuntimeError("No vectors were added to the index")
    except Exception as e:
        print(f"Error during index creation: {e}")
        return np.array([]), np.array([])
    
    # Search for nearest neighbors
    k = 2  # Get 2 nearest neighbors for ratio test
    distances, indices = index.search(query_features, k)
 
    #Apply Lowe's ratio test (modified for inner product where larger is better)
    good_matches = []
    good_distances = []
    for query_idx, (dist1, dist2) in enumerate(distances):
        if dist1 > threshold * dist2 and dist1 != dist2:  # Added check for equality
            db_idx = indices[query_idx][0]
            good_matches.append((db_idx, query_idx))
            good_distances.append(dist1)

    return np.array(good_matches).astype('int'), np.array(good_distances)

def cosine_feature_matching(features1, features2, threshold=0.8):
    # Check for zero-norm features and handle them
    norms1 = np.linalg.norm(features1, axis=1)
    norms2 = np.linalg.norm(features2, axis=1)
    features1 = features1 / norms1[:, np.newaxis] if np.all(norms1) else features1
    features2 = features2 / norms2[:, np.newaxis] if np.all(norms2) else features2
    
    # Compute similarity matrix
    similarity_matrix = np.dot(features2, features1.T)
    
    # Get top 2 matches for each query feature
    top2_similarities = np.partition(-similarity_matrix, 1, axis=1)[:, :2] * -1
    
    # Apply ratio test
    good_matches = []
    good_distances = []
    for i, (sim1, sim2) in enumerate(top2_similarities):
        
        if sim1 >= threshold * sim2:  # Corrected condition
            good_matches.append((i, np.argmax(similarity_matrix[i])))
            good_distances.append(sim1)
    print(sim1, sim2)      
    return np.array(good_matches), np.array(good_distances)    



def filter_matches_with_ransac(keypoints1, keypoints2, matches, threshold=3.0):
    """
    Filter feature matches using RANSAC to remove outliers.
    
    Parameters:
    keypoints1: List of keypoints from first image
    keypoints2: List of keypoints from second image
    matches: List of DMatch objects containing the matches between keypoints
    threshold: RANSAC threshold for considering a point as an inlier
    
    Returns:
    filtered_matches: List of DMatch objects containing only inlier matches
    """
    
    # Extract matched keypoints
    src_pts = np.float32([keypoints1[m[0]] for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m[1]] for m in matches]).reshape(-1, 1, 2)
    
    # Find homography using RANSAC
    homography_matrix, mask = cv2.findHomography(
        src_pts, 
        dst_pts, 
        cv2.RANSAC,
        threshold
    )
    
    # Convert mask to boolean array
    mask = mask.ravel().tolist()
    
    # Filter matches using the mask
    filtered_matches = [m for i, m in enumerate(matches) if mask[i]]
    
    return filtered_matches



if __name__ == "__main__":
    
    
   
    
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
 
    #vis_key_1 = visualize_keypoints(img1, img0_feature['keypoints'] )
    
    # get similarities
    matches, distances = improved_feature_matching(img1_feature['descriptors'], img2_feature['descriptors'], threshold=0.6)
    # use RANSAC to discard outliers
    filtered_matches = filter_matches_with_ransac(img1_feature['keypoints'], img2_feature['keypoints'], matches, threshold=5.0)
    #plot final result
    plot_keypoint_matches(img1, img2, img1_feature['keypoints'], img2_feature['keypoints'], filtered_matches, output_path='./src/matches.png')
    
    
