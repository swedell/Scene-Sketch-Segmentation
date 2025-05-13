import torch
from torch.utils.data import Dataset
import os
import numpy as np
import json
from PIL import Image,ImageOps
from torchvision import transforms
from torchvision.transforms.functional import rotate,crop,resize


class fscoco_train(Dataset):
    def __init__(self, root="DATA/new_train_data_more", transform=None, augment=False, SKETCH_SIZE=512):
        self.root = root
        self.transform = transform
        self.augment = augment
        
        # Initialize directories for raster sketches and vector strokes
        self.sketch_dir = os.path.join(root, "sketches")
        self.stroke_dir = os.path.join(root, "vector_sketches")
        self.text_dir = os.path.join(root,"text")

        # Initialize any augmentation transforms
        self.augmentation = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomCrop(450),
            transforms.Resize((SKETCH_SIZE, SKETCH_SIZE))
        ])
        
        # Prepare lists for all sketch and vector files
        self.sketch_files = []
        self.stroke_files = {}
        self.caption_files = {}

        for dirpath, _, filenames in os.walk(self.text_dir):
            for filename in filenames:
                base_name = os.path.splitext(filename)[0]
                self.caption_files[base_name] = os.path.join(dirpath, filename)  # Fill with path

        # Fill in sketch files and initialize stroke and caption file paths
        for dirpath, _, filenames in os.walk(self.sketch_dir):
            for filename in filenames:
                if filename.endswith('.jpg'):  # Assuming sketches are .png files
                    full_path = os.path.join(dirpath, filename)
                    self.sketch_files.append(full_path)

                    # Extract the base name (without extension) to match with vectors and captions
                    base_name = os.path.splitext(filename)[0]
                    self.stroke_files[base_name] = None  # Initialize with None, will fill with path later

        # Match strokes to sketches
        for dirpath, _, filenames in os.walk(self.stroke_dir):
            for filename in filenames:
                if filename.endswith('.npy'):  # Assuming vectors are .npy files
                    full_path = os.path.join(dirpath, filename)
                    base_name = os.path.splitext(filename)[0]
                    if base_name in self.stroke_files:
                        self.stroke_files[base_name] = full_path
        print(f"Found {len(self.sketch_files)} sketches, {len(self.stroke_files)} strokes, and {len(self.caption_files)} captions.")

    def __len__(self):
        return len(self.sketch_files)

    def __getitem__(self, index):
        # Load raster sketch image
        sketch_path = self.sketch_files[index]
        sketch = Image.open(sketch_path).convert("RGB")

        # Apply augmentations if enabled
        if self.augment:
            sketch_aug = ImageOps.invert(sketch)
            sketch = self.augmentation(sketch_aug)
            sketch = ImageOps.invert(sketch)
        
        # Apply any additional transformations
        if self.transform:
            sketch = self.transform(sketch)

        # Extract base name from sketch file to find corresponding vector file and caption
        base_name = os.path.splitext(os.path.basename(sketch_path))[0]
        stroke_path = self.stroke_files.get(base_name)
        caption_path = self.caption_files.get(base_name)

        # Load stroke data if it exists
        if stroke_path and os.path.exists(stroke_path):
            stroke_data = np.load(stroke_path, allow_pickle=True)
            stroke_data = torch.tensor(stroke_data, dtype=torch.float32)
            if stroke_data.dim() == 4:
                stroke_data = stroke_data.squeeze(-1) 
        else:
            stroke_data = torch.tensor([])  # Return an empty tensor if stroke data is missing

        # Load caption if it exists
        if caption_path and os.path.exists(caption_path):
            with open(caption_path, "r") as f:
                caption = f.read()
        else:
            caption = ""

        return sketch, stroke_data, caption 

class fscoco_test(Dataset):
    def __init__(self,root= "DATA/test"):
        self.root = root
        self.img_dir = os.path.join(root,"images")
        self.text_dir = os.path.join(root,"captions")
        self.stroke_dir = os.path.join(root,"vector_sketches")
        self.label_dir = os.path.join(root,"classes")
        
        # Get list of subdirectories for images and captions
        self.img_files = sorted(os.listdir(self.img_dir))
        self.txt_files = sorted(os.listdir(self.text_dir))
        self.strokes_files = sorted(os.listdir(self.stroke_dir))
        self.label_files = sorted(os.listdir(self.label_dir))
        print(self.img_dir,self.stroke_dir)

    def __len__(self):
        return len(self.strokes_files)
    
    def __getitem__(self,index):
        
        img_path = os.path.join(self.img_dir,self.img_files[index])
        strokes_path = os.path.join(self.stroke_dir,self.strokes_files[index])
        classes_path = os.path.join(self.label_dir,self.label_files[index])
        with open(classes_path,"r") as f:
            classes = json.load(f)
        pen_state = np.load(strokes_path,allow_pickle=True) # (n,3) array, where n is the number of pen states 
        text_path = os.path.join(self.text_dir,self.txt_files[index])
        
        with open(text_path,"r") as f:
            caption = f.read()
        print('item',pen_state,classes, caption,img_path)
        return pen_state,classes, caption,img_path
