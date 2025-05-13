import matplotlib.pyplot as plt
import os
from dataset import fscoco_train

# Initialize the dataset
dataset = fscoco_train(root="DATA/train_subset", transform=None, augment=False, SKETCH_SIZE=512)

# Print the total number of samples
print(f"Total number of samples in dataset: {len(dataset)}")

# Directory to save the images
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# Fetch a sample from the dataset
for i in range(len(dataset)):  # Loop over all samples to avoid index errors
    try:
        sketch, stroke_data, caption = dataset[i]  # Load the i-th sample

        # Save the sketch image
        plt.imshow(sketch)
        plt.title(f"Sample {i+1} - Sketch Image")
        plt.savefig(os.path.join(output_dir, f"sample_{i+1}_sketch.png"))
        plt.close()  # Close the plot to avoid overlapping plots

        # Print stroke data shape and a sample of its content
        if stroke_data is not None:
            print(f"Sample {i+1} - Stroke Data Shape: {stroke_data.shape}")
            print(f"Sample {i+1} - Stroke Data: {stroke_data}")

        # Print the caption text
        print(f"Sample {i+1} - Caption: {caption}")
    
    except IndexError as e:
        print(f"IndexError: {e} at index {i}")
        break  # Stop if there is an IndexError
