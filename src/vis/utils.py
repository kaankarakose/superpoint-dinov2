import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go



import cv2



def plot_keypoint_matches(img1, img2, keypoints1, keypoints2, matches, output_path='matches.png', max_matches=-1):
    """
    Plot and save keypoint matches between two images using matplotlib.
    
    Parameters:
    -----------
    img1: numpy.ndarray
        First input image
    img2: numpy.ndarray
        Second input image
    keypoints1: numpy.ndarray
        Keypoints coordinates from first image as numpy array of shape (N, 2)
    keypoints2: numpy.ndarray
        Keypoints coordinates from second image as numpy array of shape (N, 2)
    matches: numpy.ndarray
        Array of match indices with shape (N, 2) where N is number of matches
        Each row contains [index_in_kp1, index_in_kp2]
    output_path: str
        Path where to save the output visualization
    max_matches: int
        Maximum number of matches to draw (to avoid cluttering)
    """
    # Create figure and axis
    plt.figure(figsize=(20, 10))
    
    # Get dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Create concatenated image
    combined_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    combined_img[:h1, :w1] = img1
    combined_img[:h2, w1:w1+w2] = img2
    
    # Display the combined image
    plt.imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
    
    # Limit number of matches to display
    matches = matches[:max_matches]
    
    # Draw matches
    for kp1, kp2 in matches:
        # Get keypoints
        pt_1 = keypoints1[kp1]
        pt_2 = keypoints2[kp2]
        
        # Adjust query point x-coordinate to account for image concatenation
        pt_2_adjusted = (pt_2[0] + w1, pt_2[1])
        
        # Draw connecting line
        plt.plot([pt_1[0], pt_2_adjusted[0]], 
                [pt_1[1], pt_2_adjusted[1]], 
                'c-', linewidth=1, alpha=0.6)
        
        # Draw keypoints
        plt.plot(pt_1[0], pt_1[1], 'ro', markersize=4)
        plt.plot(pt_2_adjusted[0], pt_2_adjusted[1], 'ro', markersize=4)
    
    # Remove axes
    plt.axis('off')
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    return combined_img


def visualize_keypoints(image, keypoints):
    # Assuming keypoints are in (x,y) format
    # Make a copy of the image to draw on
    vis_image = image.copy()
    # Draw each keypoint
    for point in keypoints:
        x, y = int(point[0]), int(point[1])
        cv2.circle(vis_image, (x, y), radius=2, color=(0, 255, 0), thickness=-1)
    
    return vis_image




