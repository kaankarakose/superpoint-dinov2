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
                'c-', linewidth=0.5, alpha=0.3)
        
        # Draw keypoints
        plt.plot(pt_1[0], pt_1[1], 'ro', markersize=2)
        plt.plot(pt_2_adjusted[0], pt_2_adjusted[1], 'ro', markersize=2)
    
    # Remove axes
    plt.axis('off')
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    return combined_img









def get_top_k_matches(similarities, k=5):
    """
    Get top k matches for each query.
    
    Parameters:
    -----------
    similarities : np.ndarray
        Similarity matrix of shape (n_query, n_db)
    k : int
        Number of top matches to return
    
    Returns:
    --------
    top_k_indices : np.ndarray
        Indices of top k matches for each query (n_query, k)
    top_k_similarities : np.ndarray
        Similarity scores for top k matches (n_query, k)
    """
    top_k_similarities, top_k_indices = torch.topk(torch.from_numpy(similarities), k)
    return top_k_indices.numpy(), top_k_similarities.numpy()

def calculate_similarities(db_features, query_features, metric='cosine', batch_size=128):
    """
    Calculate similarities between database and query vectors using specified metric.
    
    Parameters:
    -----------
    db_features : np.ndarray or torch.Tensor
        Database features of shape (n_db, dim)
    query_features : np.ndarray or torch.Tensor
        Query features of shape (n_query, dim)
    metric : str
        Similarity metric to use ('cosine' or 'dot')
    batch_size : int
        Batch size for processing to manage memory usage
    
    Returns:
    --------
    similarities : np.ndarray
        Similarity matrix of shape (n_query, n_db)
    """
    # Convert to torch tensors if needed
    if isinstance(db_features, np.ndarray):
        db_features = torch.from_numpy(db_features)
    if isinstance(query_features, np.ndarray):
        query_features = torch.from_numpy(query_features)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    db_features = db_features.to(device)
    
    n_db = db_features.shape[0]
    n_query = query_features.shape[0]
    similarities = torch.zeros((n_query, n_db), device=device)
    
    # Process in batches to manage memory
    for i in range(0, n_query, batch_size):
        batch_query = query_features[i:i+batch_size].to(device)
        
        if metric == 'cosine':
            # Normalize vectors for cosine similarity
            batch_query = F.normalize(batch_query, p=2, dim=1)
            db_normalized = F.normalize(db_features, p=2, dim=1)
            
            # Calculate similarity
            batch_similarities = torch.mm(batch_query, db_normalized.t())
            
        elif metric == 'dot':
            # Simple dot product
            batch_similarities = torch.mm(batch_query, db_features.t())
        
        similarities[i:i+batch_size] = batch_similarities
    
    return similarities.cpu().numpy()


def plot_similarity_heatmap(similarities, figsize=(12, 8), sample_size=None):
    """
    Create a heatmap visualization of the similarity matrix.
    
    Parameters:
    -----------
    similarities : np.ndarray
        Similarity matrix of shape (n_query, n_db)
    figsize : tuple
        Figure size for the plot
    sample_size : int or None
        If provided, randomly sample this many queries and database items
    """
    if sample_size is not None:
        n_query, n_db = similarities.shape
        query_idx = np.random.choice(n_query, min(sample_size, n_query), replace=False)
        db_idx = np.random.choice(n_db, min(sample_size, n_db), replace=False)
        similarities_sample = similarities[query_idx][:, db_idx]
    else:
        similarities_sample = similarities

    plt.figure(figsize=figsize)
    sns.heatmap(similarities_sample, cmap='viridis', center=0)
    plt.title('Similarity Matrix Heatmap')
    plt.xlabel('Database Items')
    plt.ylabel('Query Items')
    plt.savefig("heatmap.png")

def plot_similarity_distribution(similarities):
    """
    Plot the distribution of similarity scores.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(similarities.flatten(), bins=50, density=True)
    plt.title('Distribution of Similarity Scores')
    plt.xlabel('Similarity Score')
    plt.ylabel('Density')
    plt.savefig("distribution.png")

def visualize_embeddings_tsne(query_features, db_features, perplexity=30):
    """
    Visualize both query and database features in 2D using t-SNE.
    
    Parameters:
    -----------
    query_features : np.ndarray
        Query feature vectors
    db_features : np.ndarray
        Database feature vectors
    perplexity : int
        t-SNE perplexity parameter
    """
    # Combine features
    combined_features = np.vstack([query_features, db_features])
    
    # Create labels
    labels = ['Query'] * len(query_features) + ['Database'] * len(db_features)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embedded_features = tsne.fit_transform(combined_features)
    
    # Create interactive scatter plot with plotly
    df = {
        'x': embedded_features[:, 0],
        'y': embedded_features[:, 1],
        'Type': labels
    }
    
    fig = px.scatter(df, x='x', y='y', color='Type',
                    title='t-SNE Visualization of Query and Database Features',
                    labels={'x': 't-SNE Component 1', 'y': 't-SNE Component 2'})
    fig.write_image('tsne.png')

def plot_top_k_similarities_distribution(similarities, k=5):
    """
    Plot distribution of top-k similarity scores for each query.
    
    Parameters:
    -----------
    similarities : np.ndarray
        Similarity matrix
    k : int
        Number of top matches to consider
    """
    top_k_scores = -np.sort(-similarities, axis=1)[:, :k]  # Get top k scores for each query
    
    plt.figure(figsize=(10, 6))
    positions = np.arange(1, k + 1)
    plt.boxplot([top_k_scores[:, i] for i in range(k)], positions=positions)
    
    plt.title(f'Distribution of Top-{k} Similarity Scores')
    plt.xlabel('Rank')
    plt.ylabel('Similarity Score')
    plt.savefig('k_similarities_distribution.png')