# assign_noise_images.py

import os
import pickle
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from shutil import copy2
from transformers import AutoImageProcessor, AutoModel
import csv
from sklearn.decomposition import PCA
import hdbscan

# Configuration
normalized_embeddings_path = "/home/azaa/autolabel_workspace/AutoLabel/snacks_noise_normalized_embeddings.npy"  # Path to normalized embeddings
cluster_data_file = "noise_cluster_data.pkl"                         # Path to save/load cluster data
noise_image_dir = "/home/azaa/autolabel_workspace/AutoLabel/datasets_autolabel/reoutput/snacks_Recognition_All_noise_PD_clustered/noise"                      # Replace with your noise folder path
output_dir = "/home/azaa/autolabel_workspace/datasets_autolabel/reoutput/snacks_noise2_reclustered"                          # Replace with your desired output path
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
similarity_threshold = 0.85                                     # Threshold for assigning clusters
batch_size = 16                                                 # Batch size for processing images
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# Step 1: Load Normalized Embeddings
print(f"Loading normalized embeddings from '{normalized_embeddings_path}'...")
normalized_embeddings = np.load(normalized_embeddings_path)
print(f"Shape of normalized_embeddings: {normalized_embeddings.shape}")

# Step 2: Dimensionality Reduction with PCA
print("Applying PCA for dimensionality reduction...")
pca = PCA(n_components=15, random_state=42)
reduced_embeddings = pca.fit_transform(normalized_embeddings)
print(f"Shape of reduced_embeddings: {reduced_embeddings.shape}")

# Step 3: Clustering with HDBSCAN
print("Clustering with HDBSCAN...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=10, metric='euclidean')
labels = clusterer.fit_predict(reduced_embeddings)
print(f"Number of clusters found (excluding noise): {len(set(labels)) - (1 if -1 in labels else 0)}")

# Step 4: Compute Cluster Centroids
print("Computing cluster centroids...")
unique_labels = set(labels)
unique_labels.discard(-1)  # Remove noise label if present

# Initialize dictionary to store centroids
cluster_centroids = {}
for label in unique_labels:
    cluster_points = normalized_embeddings[labels == label]
    centroid = cluster_points.mean(axis=0)
    centroid_normalized = centroid / np.linalg.norm(centroid)  # Normalize the centroid
    cluster_centroids[label] = centroid_normalized

print(f"Computed centroids for {len(cluster_centroids)} clusters.")

# Step 5: (Optional) Save Cluster Data
print(f"Saving cluster data to '{cluster_data_file}'...")
cluster_data = {
    'cluster_centroids': cluster_centroids,
    'labels': labels
}
with open(cluster_data_file, 'wb') as f:
    pickle.dump(cluster_data, f)
print("Cluster data saved successfully.")

# Step 6: Load Cluster Data (if needed in future runs)
# Uncomment the following lines if you want to load cluster data instead of recomputing
"""
print(f"Loading cluster data from '{cluster_data_file}'...")
with open(cluster_data_file, 'rb') as f:
    cluster_data = pickle.load(f)

labels = cluster_data['labels']
cluster_centroids = cluster_data['cluster_centroids']
print("Cluster data loaded successfully.")
"""

# Step 7: Prepare for Assigning Noise Images
# Convert cluster_centroids to NumPy array for similarity computation
cluster_labels_sorted = sorted(cluster_centroids.keys())
cluster_centroids_array = np.vstack([cluster_centroids[label] for label in cluster_labels_sorted])
print(f"Cluster centroids array shape: {cluster_centroids_array.shape}")

# (Optional) If you have category labeling data, load it here
# Assuming category labeling data is part of cluster_data.pkl
use_labeling = False
if 'category_centroids' in cluster_data:
    use_labeling = True
    category_centroids = cluster_data['category_centroids']
    category_names = cluster_data['category_names']
    cluster_category_mapping = cluster_data['cluster_category_mapping']
    cluster_similarity_scores = cluster_data['cluster_similarity_scores']
    print("Category labeling data found.")
else:
    print("No category labeling data found. Proceeding without category labels.")

# Initialize DINOv2 Model and Processor for Feature Extraction
print("Loading DINOv2 model and processor...")
processor_dino = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
model_dino = AutoModel.from_pretrained('facebook/dinov2-large').to(device)
model_dino.eval()  # Set model to evaluation mode

# Function to extract embeddings
def extract_embeddings(image_paths, processor, model, device, batch_size=16):
    embeddings = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting embeddings for noise images"):
        batch_paths = image_paths[i:i+batch_size]
        images = []
        for path in batch_paths:
            try:
                image = Image.open(path).convert("RGB")
                images.append(image)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
        if images:
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)
    if embeddings:
        return np.vstack(embeddings)
    else:
        return np.array([])

# Step 8: Collect Noise Image Paths
print("Collecting noise image paths...")
noise_image_paths = []
for fname in os.listdir(noise_image_dir):
    if fname.lower().endswith(image_extensions):
        noise_image_paths.append(os.path.join(noise_image_dir, fname))

print(f"Total noise images found: {len(noise_image_paths)}")

if not noise_image_paths:
    print("No noise images found. Exiting.")
    exit()

# Step 9: Extract and Normalize Embeddings for Noise Images
print("Extracting embeddings for noise images...")
noise_embeddings = extract_embeddings(noise_image_paths, processor_dino, model_dino, device, batch_size)

if noise_embeddings.size == 0:
    print("No valid embeddings extracted from noise images. Exiting.")
    exit()

noise_embeddings_normalized = normalize(noise_embeddings)
print(f"Shape of normalized noise embeddings: {noise_embeddings_normalized.shape}")

# Step 10: Compute Cosine Similarity and Assign Clusters
print("Computing cosine similarity between noise images and cluster centroids...")
similarity_matrix = cosine_similarity(noise_embeddings_normalized, cluster_centroids_array)

# Determine best cluster for each noise image
best_matches = similarity_matrix.argmax(axis=1)
best_scores = similarity_matrix.max(axis=1)

# Assign clusters based on similarity threshold
assignments = []
for match, score in zip(best_matches, best_scores):
    if score >= similarity_threshold:
        assignments.append(match)
    else:
        assignments.append('uncategorized')

# Step 11: Assign Noise Images to Cluster Folders and Log Assignments
print("Assigning noise images to clusters and logging assignments...")

# Initialize list to store assignment records
assignment_records = []

for idx, assignment in enumerate(assignments):
    src_path = noise_image_paths[idx]
    image_name = os.path.basename(src_path)
    similarity_score = best_scores[idx]
    
    if assignment != 'uncategorized':
        cluster_label = cluster_labels_sorted[assignment]  # Cluster label (e.g., 0, 1, 2, ...)
        if use_labeling:
            assigned_category = cluster_category_mapping.get(cluster_label, 'uncategorized')
            cluster_dir = os.path.join(output_dir, f'cluster_{cluster_label}_{assigned_category.replace(" ", "_")}')
        else:
            assigned_category = 'N/A'  # Not applicable if no labeling
            cluster_dir = os.path.join(output_dir, f'cluster_{cluster_label}')
    else:
        cluster_label = 'uncategorized'
        assigned_category = 'N/A' if not use_labeling else 'uncategorized'
        cluster_dir = os.path.join(output_dir, 'uncategorized_noise')
    
    os.makedirs(cluster_dir, exist_ok=True)
    dst_path = os.path.join(cluster_dir, image_name)
    copy2(src_path, dst_path)
    
    # Append assignment record
    assignment_records.append({
        'image_name': image_name,
        'source_path': src_path,
        'assigned_cluster': cluster_label,
        'assigned_category': assigned_category,
        'similarity_score': similarity_score
    })

# Step 12: Save Assignment Records to a CSV File
print("Saving assignment records to CSV file...")

# Define the path for the assignment log file
assignment_log_file = os.path.join(output_dir, 'noise_image_assignments.csv')

# Define CSV headers
csv_headers = ['image_name', 'source_path', 'assigned_cluster', 'assigned_category', 'similarity_score']

# Write to CSV
with open(assignment_log_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
    writer.writeheader()
    for record in assignment_records:
        writer.writerow(record)

print(f"Assignment log saved successfully to '{assignment_log_file}'.")
print("Noise image assignment completed successfully.")
