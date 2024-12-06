from transformers import Trainer
import torch
import numpy as np

from PIL import Image
from io import BytesIO


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        image = row['image']
        label = row['target']

        # Check if the image is in bytes, and convert to PIL if needed
        if isinstance(image, bytes):
            image = Image.open(BytesIO(image)).convert("RGB")

        processed = self.processor(
            text=None,
            images=image,
            padding="max_length",
            return_tensors="pt"
        )
        return processed["pixel_values"].squeeze(0), label


class EarlyStopping:
    def __init__(self, patience=5, delta=0.001, verbose=False):
        """Early stopping flag for training.

        Parameters:
            patience (int): How many epochs to wait after the last improvement.
            delta (float): Minimum change to consider an improvement.
            verbose (bool): Print messages if True.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def evaluate(peft_model, valid_loader, criterion, device):
    """Evaluate the model on the validation set."""
    peft_model.eval()  # Set the model to evaluation mode
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for pixel_values, labels in valid_loader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = peft_model(pixel_values=pixel_values)
            loss = criterion(outputs.logits, labels)
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = correct / total
    val_loss = val_loss / len(valid_loader)
    return val_loss, val_accuracy