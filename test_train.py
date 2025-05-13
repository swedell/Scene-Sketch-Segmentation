import torch
from torch import nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from dataset import fscoco_train
from utils import setup
from train import VisionStrokeTransformer

# Constants
BICUBIC = InterpolationMode.BICUBIC

def test_single_image_vector(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define preprocessors
    preprocess_no_T = Compose([
        Resize((224, 224), interpolation=BICUBIC),
        ToTensor(),  # Converts PIL Image to Tensor
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    # Initialize the model
    model = VisionStrokeTransformer(d_model=512, nhead=8, num_layers=6, dim_feedforward=2048).to(device)
    model.eval()  # Set model to evaluation mode
    print("VST Model initialized successfully")

    # Load a single sample from the dataset
    dataset = fscoco_train(transform=preprocess_no_T, augment=False)
    single_sample = dataset[0]  # Load the first sample

    # Unpack the sample
    sketch, vector_data, caption = single_sample
    print(f"Single Sample - Sketch Shape: {sketch.shape}, Vector Data Shape: {vector_data.shape}, Caption: {caption}")

    # Move data to the appropriate device
    sketch = sketch.unsqueeze(0).to(device)  # Add batch dimension
    vector_data = vector_data.unsqueeze(0).to(device)  # Add batch dimension

    # Run a forward pass through the model
    with torch.no_grad():
        output = model(sketch, vector_data)
        print(f"Model Output: {output}")

if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    cfg = setup(args)

    # Test the model with a single image and vector
    test_single_image_vector(cfg)
