import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from sklearn.preprocessing import normalize


class FeatureEmbedder:
    def __init__(self, model, batch_size, model_processor, device = "cuda"):
        self.model = model
        self.batch_size = batch_size
        self.model_processor = model_processor
        self.output_dir = None
        self.device=device
        
    def generate_embeddings(self, source_path):
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        image_paths = []
        for root, _, files in os.walk(source_path):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_paths.append(os.path.join(root, file))

        embeddings = []
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="Extracting image features"):
            batch_paths = image_paths[i:i+self.batch_size]
            images = []
            for image_path in batch_paths:
                try:
                    image = Image.open(image_path).convert("RGB")
                    images.append(image)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    continue
            if images:
                inputs = self.model_processor(images=images, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(batch_embeddings)

        embeddings = np.vstack(embeddings)

        normalized_embeddings = normalize(embeddings)
        np.save(f"{os.path.basename(os.path.normpath(source_path))}_normalized_embeddings.npy", normalized_embeddings)
        return normalized_embeddings