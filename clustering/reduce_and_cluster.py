import argparse
import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
import numpy as np
from shutil import copy2
import hdbscan
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import umap.umap_ as umap
from sklearn.cluster import KMeans

class ReduceandCluster:

    def reduce_and_cluster(self, normalized_embeddings, best_params):
        red_type = best_params["red_type"]
        clustering_type = best_params["clustering_type"]
        n_components = best_params["n_components"]
        n_neighbors = best_params["n_neighbors"]
        min_distance = best_params["min_distance"]
        min_cluster_size = best_params["min_cluster_size"]
        min_samples = best_params["min_samples"]
        n_clusters = best_params["n_clusters"]

        if red_type == "pca":
            pca = PCA(n_components=n_components, random_state=42)
            reduced_embeddings = pca.fit_transform(normalized_embeddings)
        else:
            umap_reducer = umap.UMAP(n_neighbors=n_neighbors, min_distance=min_distance, random_state=42)
            reduced_embeddings = umap_reducer.fit_transform(normalized_embeddings)

        
        if clustering_type == "hdbscan":

            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,min_samples=min_samples, random_state=42)
            labels = clusterer.fit_predict(reduced_embeddings)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(reduced_embeddings)
        
         np.save("labels.npy", labels) 

        return reduced_embeddings
    