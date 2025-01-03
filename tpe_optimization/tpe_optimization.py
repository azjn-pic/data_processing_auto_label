import json
import optuna
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
# import hdbscan
# import umap.umap_ as umap
import argparse
import torch
from cuml.cluster import HDBSCAN
from cuml.decomposition import PCA as cumlPCA
from cuml.manifold import UMAP



class TPEOptimizer:
    def __init__(self, device="cuda"):
        """
        Initialize any variables/resources here.
        """
        self.device = device

    # Define the objective functions
    def objective_umap_hdbscan(trial, embeddings):
        # Suggest UMAP parameters
        n_neighbors = trial.suggest_int('n_neighbors', 5, 50)
        min_dist = trial.suggest_float('min_dist', 0.0, 0.5)
        n_components = trial.suggest_int('n_components', 10, 30)

        # Suggest HDBSCAN parameters
        min_cluster_size = trial.suggest_int('min_cluster_size', 5, 30)
        min_samples = trial.suggest_int('min_samples', 1, 10)

        # UMAP dimensionality reduction
        umap_reducer = UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            random_state=42
        )
        reduced_embeddings = umap_reducer.fit_transform(embeddings)

        # HDBSCAN clustering
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        labels = clusterer.fit_predict(reduced_embeddings)

        if len(set(labels.tolist()) - {-1}) > 1:  # Convert labels to a list for set operations
            non_noise_mask = labels != -1
            reduced_embeddings_cpu = reduced_embeddings[non_noise_mask].get()  # Move to CPU
            labels_cpu = labels[non_noise_mask].get()  # Move to CPU
            score = silhouette_score(reduced_embeddings_cpu, labels_cpu)
        else:
            score = -1.0
        return score

    def objective_pca_hdbscan(trial, embeddings):
        # Suggest PCA parameters
        n_components = trial.suggest_int('n_components', 5, 50)

        # Suggest HDBSCAN parameters
        min_cluster_size = trial.suggest_int('min_cluster_size', 5, 30)
        min_samples = trial.suggest_int('min_samples', 1, 10)

        # PCA dimensionality reduction
        # pca = PCA(n_components=n_components, random_state=42)
        # reduced_embeddings = pca.fit_transform(data)

        # Inside objective_pca_hdbscan
        pca = cumlPCA(n_components=n_components, random_state=42)
        reduced_embeddings = pca.fit_transform(embeddings)


        # HDBSCAN clustering
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        labels = clusterer.fit_predict(reduced_embeddings)

        # # Silhouette Score
        # if len(set(labels) - {-1}) > 1:
        #     non_noise_mask = labels != -1
        #     score = silhouette_score(reduced_embeddings[non_noise_mask], labels[non_noise_mask])
        # else:
        #     score = -1.0
        # return score
        # Silhouette Score
        # Silhouette Score
        if len(set(labels.tolist()) - {-1}) > 1:  # Convert labels to a list for set operations
            non_noise_mask = labels != -1
            reduced_embeddings_cpu = reduced_embeddings[non_noise_mask].get()  # Move to CPU
            labels_cpu = labels[non_noise_mask].get()  # Move to CPU
            score = silhouette_score(reduced_embeddings_cpu, labels_cpu)
        else:
            score = -1.0
        return score

        

    def objective_pca_kmeans(trial, embeddings):
        # Suggest PCA parameters
        n_components = trial.suggest_int('n_components', 2, 100)

        # PCA dimensionality reduction
        pca = PCA(n_components=n_components, random_state=42)
        reduced_embeddings = pca.fit_transform(embeddings)

        # K-Means clustering
        k = 35  # Fixed number of clusters
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(reduced_embeddings)

        # Silhouette Score
        score = silhouette_score(reduced_embeddings, labels)
        return score

    def run_optimization(self, embeddings, dim_reduction, clustering_method, n_trials):
        """
        This method:
          1. Chooses the correct objective function based on config
          2. Creates and runs the Optuna study
          3. Returns the best parameters and best score
        """
        # Select objective function
        if dim_reduction.upper() == "umap" and clustering_method.upper() == "hdbscan":
            objective_fn = lambda trial: self.objective_umap_hdbscan(trial, embeddings)
            output_filename = "best_params_umap_hdbscan.json"

        elif dim_reduction.upper() == "pca" and clustering_method.upper() == "hdbscan":
            objective_fn = lambda trial: self.objective_pca_hdbscan(trial, embeddings)
            output_filename = "best_params_pca_hdbscan.json"

        elif dim_reduction.upper() == "pca" and clustering_method.upper() == "kmeans":
            objective_fn = lambda trial: self.objective_pca_kmeans(trial, embeddings)
            output_filename = "best_params_pca_kmeans.json"
        else:
            raise ValueError(f"Invalid combination: {dim_reduction} + {clustering_method}")

        # Create Optuna study and run optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_fn, n_trials=n_trials)

        best_params = study.best_params
        best_score = study.best_value

        # # Optionally save results to JSON
        # info_to_save = {
        #     "method": {
        #         "dimensionality_reduction": dim_reduction,
        #         "clustering": clustering_method
        #     },
        #     "best_params": best_params,
        #     "best_score": best_score
        # }
        # with open(output_filename, "w") as f:
        #     json.dump(info_to_save, f, indent=4)
        # print(f"Best parameters saved to {output_filename}")

        final_params = {
        "red_type": dim_reduction,        # Set from arguments
        "clustering_type": clustering_method,
        "n_components": None,
        "n_neighbors": None,
        "min_distance": None,
        "min_cluster_size": None,
        "min_samples": None,
        "n_clusters": None,
        "similarity_threshold": None
        }

        # 2. For any keys that appear in raw_optuna_params, overwrite
        #    For example, if we did tune "n_components", put that value in final_params
        for key, val in best_params.items():
            if key in final_params:
                final_params[key] = val
            else:
                # In case your TPE objective has extra keys 
                # that aren't in default_params structure, ignore or store them:
                pass

        return final_params
