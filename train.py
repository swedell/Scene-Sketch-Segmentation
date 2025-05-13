import torch
from torch import nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomRotation
from dataset import fscoco_train  # Assuming your dataset is defined here
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import clip  # Assuming you're using OpenAI's CLIP model
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for plotting (headless mode)


def save_checkpoint(model, optimizer, epoch, filename='model_checkpoint.pth'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at {filename}")
def load_checkpoint(model, optimizer, filename='model_checkpoint.pth'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {filename}, starting from epoch {start_epoch}")
    return model, optimizer, start_epoch



# Define your custom collate function
def custom_collate_fn(batch):
    sketches, vector_data, masks, captions = zip(*batch)
    sketches = torch.stack(sketches)
    masks = torch.stack(masks)
    max_len = max(vd.size(0) for vd in vector_data)
    vector_data = torch.stack([pad_vector_data(vd, max_len) for vd in vector_data])
    return sketches, vector_data, masks, captions

def pad_vector_data(vector_data, max_len):
    padded = torch.zeros((max_len, 3))  # Assuming vector data has shape [seq_len, 3]
    padded[:vector_data.size(0), :] = vector_data
    return padded

class CombinedBCEAndDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedBCEAndDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()  # Logits-based BCE loss

    def forward(self, outputs, masks):
        assert outputs.shape == masks.shape, f"Shape mismatch: {outputs.shape} vs {masks.shape}"
        bce_loss = self.bce_loss(outputs, masks)
        dice_loss_value = self.dice_loss(outputs, masks)
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss_value
        return total_loss

    def dice_loss(self, pred, target, smooth=1.):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        denominator = pred.sum() + target.sum()
        print(f"Intersection: {intersection.item()}, Denominator: {denominator.item()}")
        if denominator == 0:
            return 0.0
        dice = (2. * intersection + smooth) / (denominator + smooth)
        return 1 - dice

class VisionStrokeTransformerForSegmentation(nn.Module):
    def __init__(self, clip_model, d_model=256, nhead=4, num_layers=3, dim_feedforward=1024, dropout=0.5, experiment='baseline'):
        super(VisionStrokeTransformerForSegmentation, self).__init__()

        self.experiment = experiment

        # DeepLabV3 model with ResNet50 backbone (for raster images)
        self.deeplabv3 = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)

        # CLIP Text Encoder (frozen model for captions)
        self.clip_model = clip_model

        # Transformer for Stroke Data (for vector inputs)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.stroke_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Linear projection for vector stroke data
        self.vector_projection = nn.Linear(3, d_model)

        # Conditionally set the final convolution layer based on the experiment type
        if self.experiment == 'baseline':
            # Only raster (2048) and text (512) features in baseline
            self.fc = nn.Conv2d(2048 + 512, 1, kernel_size=1)
        else:
            # For all other experiments, include raster (2048), vector (d_model), and text (512) features
            self.fc = nn.Conv2d(2048 + d_model + 512, 1, kernel_size=1)
            self.dropout = nn.Dropout(dropout)
    def forward(self, raster_input, vector_input, caption_input):
        # Raster branch: extract features using DeepLabV3
        raster_features = self.deeplabv3.backbone(raster_input)['out']  # Raster features (2048 channels)

        # Text branch: encode captions using CLIP (512 channels)
        with torch.no_grad():
            text_embedding = self.clip_model.encode_text(caption_input)

        if self.experiment == 'baseline':
            # In the baseline experiment, only use raster and text features
            text_features = text_embedding.view(text_embedding.size(0), text_embedding.size(1), 1, 1)
            text_features = text_features.expand(-1, -1, raster_features.shape[2], raster_features.shape[3])

            # Concatenate raster and text features
            combined_features = torch.cat((raster_features, text_features), dim=1)

        elif self.experiment == 'experiment1':
            # Vector branch: project stroke data to d_model dimensions
            vector_features = self.vector_projection(vector_input)

            # Expand the vector features for concatenation
            vector_features = vector_features.sum(dim=1).view(vector_features.size(0), vector_features.size(2), 1, 1)
            vector_features = vector_features.expand(-1, -1, raster_features.shape[2], raster_features.shape[3])

            # Expand text features
            text_features = text_embedding.view(text_embedding.size(0), text_embedding.size(1), 1, 1)
            text_features = text_features.expand(-1, -1, raster_features.shape[2], raster_features.shape[3])

            # Concatenate raster, vector, and text features
            combined_features = torch.cat((raster_features, vector_features, text_features), dim=1)

        elif self.experiment == 'experiment2':
            # Experiment 2: distinct self-attention for raster and vector data
            vector_features = self.vector_projection(vector_input)
            vector_features = self.stroke_transformer(vector_features)  # No unpacking needed
            vector_features = vector_features.sum(dim=1).view(vector_features.size(0), vector_features.size(2), 1, 1)
            vector_features = vector_features.expand(-1, -1, raster_features.shape[2], raster_features.shape[3])

            text_features = text_embedding.view(text_embedding.size(0), text_embedding.size(1), 1, 1)
            text_features = text_features.expand(-1, -1, raster_features.shape[2], raster_features.shape[3])

            combined_features = torch.cat((raster_features, vector_features, text_features), dim=1)

        elif self.experiment == 'experiment3':
            # Experiment 3: cross-attention combining raster and vector
            vector_features = self.vector_projection(vector_input)
            vector_features = self.stroke_transformer(vector_features)
            vector_features = vector_features.sum(dim=1).view(vector_features.size(0), vector_features.size(2), 1, 1)
            vector_features = vector_features.expand(-1, -1, raster_features.shape[2], raster_features.shape[3])

            text_features = text_embedding.view(text_embedding.size(0), text_embedding.size(1), 1, 1)
            text_features = text_features.expand(-1, -1, raster_features.shape[2], raster_features.shape[3])

            # Cross-attention can be applied here (to improve multimodal fusion)
            combined_features = torch.cat((raster_features, vector_features, text_features), dim=1)
            combined_features = self.dropout(combined_features)
        # Pass through the final convolution layer
        output = self.fc(combined_features)

        # Upsample to match the original input size (224x224)
        output = nn.functional.interpolate(output, size=(224, 224), mode='bilinear', align_corners=False)

        return output





def compute_iou(pred_mask, true_mask, threshold=0.5, eps=1e-6):
    """
    Compute Intersection over Union (IoU) for binary segmentation.
    :param pred_mask: Predicted mask (logits or probabilities)
    :param true_mask: Ground truth mask (0 or 1)
    :param threshold: Threshold for binarizing predicted mask
    :param eps: Small epsilon to avoid division by zero
    :return: IoU score
    """
    pred_mask = (pred_mask > threshold).float()  # Apply threshold here
    assert pred_mask.shape == true_mask.shape, f"Shape mismatch: {pred_mask.shape} vs {true_mask.shape}"

    intersection = (pred_mask * true_mask).sum((1, 2))
    union = pred_mask.sum((1, 2)) + true_mask.sum((1, 2)) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

def compute_pixel_accuracy(pred_mask, true_mask):
    pred_mask = (pred_mask > 0.5).float()
    correct_pixels = (pred_mask == true_mask).float().sum()
    total_pixels = true_mask.numel()
    accuracy = correct_pixels / total_pixels
    return accuracy.item()

def plot_training_graph(epochs, train_losses, val_losses, train_ious, val_ious, train_accuracies, val_accuracies):
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_ious, label='Training IoU')
    plt.plot(epochs, val_ious, label='Validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.title('Training and Validation IoU')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_accuracies, label='Training Pixel Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Pixel Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Pixel Accuracy')
    plt.title('Training and Validation Pixel Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()

def plot_confusion_matrix(true_mask, pred_mask, num_classes=2):
    true_mask = true_mask.flatten().cpu().numpy()
    pred_mask = (pred_mask > 0.5).flatten().cpu().numpy()
    cm = confusion_matrix(true_mask, pred_mask, labels=list(range(num_classes)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig('confusion_matrix.png')
    plt.close()

def visualize_segmentation(sketch, pred_mask, true_mask=None, output_dir="segmentations", figsize=(15, 5)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    sketch_np = sketch.permute(1, 2, 0).cpu().numpy()
    pred_mask_np = pred_mask.squeeze(0).cpu().numpy()
    pred_mask_np = (pred_mask_np > 0.5).astype(np.uint8)

    pred_overlay = np.zeros_like(sketch_np)
    pred_overlay[pred_mask_np == 1] = [1.0, 0.0, 0.0]
    
    plt.figure(figsize=figsize)

    plt.subplot(1, 3, 1)
    plt.imshow(sketch_np)
    plt.title("Original Sketch")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(sketch_np, alpha=0.7)
    plt.imshow(pred_overlay, alpha=0.3)
    plt.title("Predicted Mask")
    plt.axis("off")

    if true_mask is not None:
        true_mask_np = true_mask.squeeze(0).cpu().numpy()
        true_mask_np = (true_mask_np > 0.5).astype(np.uint8)
        true_overlay = np.zeros_like(sketch_np)
        true_overlay[true_mask_np == 1] = [0.0, 1.0, 0.0]
        
        plt.subplot(1, 3, 3)
        plt.imshow(sketch_np, alpha=0.7)
        plt.imshow(true_overlay, alpha=0.3)
        plt.title("True Mask")
        plt.axis("off")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(output_dir, f"segmentation_comparison_{timestamp}.png")
    plt.savefig(save_path)
    plt.close()



def validate_with_thresholds(model, val_dataloader, device):
    """
    Validate the model using multiple thresholds and choose the best one.
    :param model: The trained model
    :param val_dataloader: DataLoader for validation data
    :param device: CUDA or CPU
    :return: Best threshold and corresponding metrics
    """
    model.eval()
    thresholds = [0.4, 0.5, 0.6]  # Test different thresholds
    best_threshold = 0.5
    best_iou = 0.0

    for threshold in thresholds:
        total_iou = 0.0
        total_val_samples = 0

        with torch.no_grad():
            for sketches, vector_data, masks, captions in val_dataloader:
                sketches = sketches.to(device)
                vector_data = vector_data.to(device)
                masks = masks.float().to(device)
                masks = masks.unsqueeze(1)

                captions = clip.tokenize(captions).to(device)

                outputs = model(sketches, vector_data, captions)
                pred_masks = torch.sigmoid(outputs)  # Get probabilities

                # Compute IoU for each threshold
                iou = compute_iou(pred_masks, masks, threshold=threshold)
                total_iou += iou
                total_val_samples += 1

        avg_iou = total_iou / total_val_samples
        print(f"Threshold: {threshold}, Validation IoU: {avg_iou:.4f}")

        # Keep track of the best threshold
        if avg_iou > best_iou:
            best_iou = avg_iou
            best_threshold = threshold

    return best_threshold, best_iou


def train_model(experiment='baseline'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 10

    preprocess_transforms = Compose([
        RandomHorizontalFlip(),
        RandomRotation(15),
        Resize((224, 224)),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    clip_model, _ = clip.load('ViT-B/32', device=device)
    model = VisionStrokeTransformerForSegmentation(clip_model=clip_model, d_model=256, nhead=4, num_layers=3, experiment=experiment).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

    criterion = CombinedBCEAndDiceLoss(bce_weight=0.5, dice_weight=0.5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    full_dataset = fscoco_train(transform=preprocess_transforms, augment=False)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

    epochs, train_losses, val_losses = [], [], []
    train_ious, val_ious = [], []
    train_accuracies, val_accuracies = [],[]

    for epoch in range(num_epochs):
        model.train()
        epoch_loss, total_iou, total_accuracy = 0.0, 0.0, 0.0

        for sketches, vector_data, masks, captions in train_dataloader:
            sketches, vector_data, masks = sketches.to(device), vector_data.to(device), masks.float().to(device)
            masks = masks.unsqueeze(1)
            captions = clip.tokenize(captions).to(device)

            outputs = model(sketches, vector_data, captions)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            if torch.isnan(loss):
                print(f"NaN loss encountered at epoch {epoch}, stopping training.")
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pred_masks = torch.sigmoid(outputs)
            iou = compute_iou(pred_masks, masks)
            accuracy = compute_pixel_accuracy(pred_masks, masks)

            total_iou += iou
            total_accuracy += accuracy

        avg_loss = epoch_loss / len(train_dataloader)
        avg_iou = total_iou / len(train_dataloader)
        avg_accuracy = total_accuracy / len(train_dataloader)

        train_losses.append(avg_loss)
        train_ious.append(avg_iou)
        train_accuracies.append(avg_accuracy)
        epochs.append(epoch + 1)

        # Validation phase
        model.eval()
        best_threshold, best_iou = validate_with_thresholds(model, val_dataloader, device)
        print(f"Best threshold: {best_threshold}, Best IoU: {best_iou:.4f}")
        val_loss, total_val_iou, total_val_accuracy = 0.0, 0.0, 0.0

        with torch.no_grad():
            for sketches, vector_data, masks, captions in val_dataloader:
                sketches, vector_data, masks = sketches.to(device), vector_data.to(device), masks.float().to(device)
                masks = masks.unsqueeze(1)
                captions = clip.tokenize(captions).to(device)

                outputs = model(sketches, vector_data, captions)
                loss = criterion(outputs, masks)

                val_loss += loss.item()
                pred_masks = torch.sigmoid(outputs)

                iou = compute_iou(pred_masks, masks)
                accuracy = compute_pixel_accuracy(pred_masks, masks)

                total_val_iou += iou
                total_val_accuracy += accuracy

                visualize_segmentation(sketches[0], pred_masks[0], true_mask=masks[0], output_dir=f"segmentations")

        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_iou = total_val_iou / len(val_dataloader)
        avg_val_accuracy = total_val_accuracy / len(val_dataloader)

        val_losses.append(avg_val_loss)
        val_ious.append(avg_val_iou)
        val_accuracies.append(avg_val_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}, Accuracy: {avg_accuracy:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}, Validation IoU: {avg_val_iou:.4f}, Validation Accuracy: {avg_val_accuracy:.4f}')
        
        scheduler.step(avg_val_loss)
    save_checkpoint(model, optimizer, epoch=num_epochs, filename="final_model_weights_experiment3.pth")
    # Plot training graph
    plot_training_graph(epochs, train_losses, val_losses, train_ious, val_ious, train_accuracies, val_accuracies)

    # Plot confusion matrix
    plot_confusion_matrix(masks[0], pred_masks[0])

if __name__ == '__main__':
    train_model(experiment='experiment3')
