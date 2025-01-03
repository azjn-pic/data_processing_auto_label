import yaml
import os
import torch
from transformers import AutoImageProcessor, AutoModel
# from data_processing.raw_data_loader import load_raw_data
# from data_processing.video_to_images import convert_video_to_images
# from noise_handling.noise_reassignment import NoiseReassigner

from detection.shelf_and_product_detection import ShelfandProductDetector
from embeddings.feature_embedding import FeatureEmbedder
from tpe_optimization.tpe_optimization import TPEOptimizer
from clustering.reduce_and_cluster import ReduceandCluster


def main():
    
    # 1. Load config
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    opt = config["opt"]
    # 2. Load raw data
    #    If raw data is video, convert it to images. If it's already images, just load them directly.
    # input_path = config["data"]["input_path"]
    # raw_data = load_raw_data(input_path)

    # 2a. If data is video, convert to images
    #     (Pseudo-check, real logic depends on config or file checks)
    # if "video" in input_path:
    #     images_path = convert_video_to_images(input_path, config["data"]["output_path"])
    # else:

    images_path = "/home/azaa/data_preprocessing_pipeline/input_path"  # Already images
    output_dir = "/home/azaa/data_preprocessing_pipeline/output_dir"

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    detector = ShelfandProductDetector(
        shelf_model=config["detection"]["shelf_detection_model_path"],
        product_model=config["detection"]["product_detection_model_path"],
        base_output_dir=output_dir
    )
    detector.detect_shelves_then_products(images_path)

    model_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
    model = AutoModel.from_pretrained('facebook/dinov2-large').to(device)

    embedder= FeatureEmbedder(batch_size=config["embedding"]["batch_size"], model_processor=model_processor, model=model)
    normalized_embeddings = embedder.generate_embeddings(source_path=images_path)

    # 5. Hyperparameter optimization with TPE
    optimizer = TPEOptimizer()

    if opt:
        best_params = optimizer.run_optimization(
            normalized_embeddings, 
            dim_reduction=config["clustering"]["default_dim_red"], 
            clustering_method=config["clustering"]["default_clustering"],
            n_trials=config["optimization"]["n_trials"])
    else:

        best_params = config["default_params"] 

    reducer_and_clusterer = ReduceandCluster()
    reduced_embeddings = reducer_and_clusterer.reduce_and_cluster(normalized_embeddings, best_params)

if __name__ == "__main__":
    main()
