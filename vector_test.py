import numpy as np
import torch
from torch import nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from stroke_transformer import StrokeTransformerEncoder

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    print("Starting script...")

    # Define transforms for raster images
    preprocess_raster = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Initialize the stroke transformer for vector data
    d_model = 256  # Embedding size for the transformer encoder
    transformer_encoder = StrokeTransformerEncoder(
        num_layers=6, d_model=d_model, nhead=8, dim_feedforward=512
    ).to(device)
    print("Transformer encoder initialized.")

    # Projection layer to match vector features to raster feature size
    projection_layer = torch.nn.Linear(d_model, 512).to(device)
    print("Projection layer initialized.")

    # Load your raster image
    image_path = "/home/swedel/Documents/disertation/newproj/Scene-Sketch-Segmentation/DATA/train_subset/sketches/6/000000035299.png"  # Replace with actual path
    image = Image.open(image_path)
    raster_input = preprocess_raster(image).unsqueeze(0).to(device)  # Shape: [1, 3, 224, 224]
    print(f"Raster input shape: {raster_input.shape}")

    # Load your vector data (assuming it's stored in a .npy file)
    vector_data_path = "/home/swedel/Documents/disertation/newproj/Scene-Sketch-Segmentation/DATA/train_subset/vector_sketches/6/000000035299.npy"  # Replace with actual path
    vector_data = np.load(vector_data_path)
    vector_input = torch.tensor(vector_data).unsqueeze(0).float().to(device)  # Shape: [1, 3196, 3]
    print(f"Vector input shape: {vector_input.shape}")

    # Process the vector data through the transformer encoder
    vector_embedding = transformer_encoder(vector_input)  # Shape: [1, 3196, 256]
    print(f"Vector embedding shape after transformer: {vector_embedding.shape}")

    # Apply mean reduction across the sequence length dimension (dim=1)
    vector_embedding_mean = vector_embedding.mean(dim=1)  # Shape should be [1, 256]
    print(f"Vector embedding shape after mean reduction: {vector_embedding_mean.shape}")

    # Project the vector embedding to the raster feature space
    vector_features = projection_layer(vector_embedding_mean)  # Shape: [1, 512]
    print(f"Vector features shape after processing: {vector_features.shape}")

    # Simulate raster features (replace with actual model processing)
    raster_features = torch.randn(1, 512).to(device)  # Simulated raster features
    print(f"Raster features shape: {raster_features.shape}")

    # Combine raster and vector features
    combined_features = torch.cat((raster_features, vector_features), dim=1)  # Shape: [1, 1024]
    print(f"Combined features shape: {combined_features.shape}")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print("Script finished.")
