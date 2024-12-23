import torch

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


def print_trainable_parameters(model) -> None:
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )


def evaluate(peft_model, valid_loader, criterion, device) -> tuple[float, float]:
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
            outputs = peft_model.vision_model(pixel_values=pixel_values)
            loss = criterion(outputs.pooler_output, labels)
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.pooler_output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = correct / total
    val_loss = val_loss / len(valid_loader)
    return val_loss, val_accuracy


class PerformanceLogger:
    def __init__(self):
        self.log_data = {
            "epoch": [],
            "batch": [],
            "batch_loss": [],
            "batch_accuracy": [],
            "avg_loss": [],
            "avg_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }

    def log_batch(self, epoch, batch, batch_loss, batch_accuracy, avg_loss, avg_accuracy):
        """Log batch-wise metrics."""
        self.log_performance(
            self,
            epoch=epoch,
            batch=batch,
            batch_loss=batch_loss,
            batch_accuracy=batch_accuracy,
            avg_loss=avg_loss,
            avg_accuracy=avg_accuracy,
            val_loss=None,  # Placeholder for validation metrics
            val_accuracy=None,
        )

    def log_epoch(self, epoch, avg_loss, avg_accuracy, val_loss, val_accuracy):
        """Log end-of-epoch metrics."""
        self.log_performance(
            self,
            epoch=epoch,
            batch=None,  # No batch info for epoch-level logging
            batch_loss=None,
            batch_accuracy=None,
            avg_loss=avg_loss,
            avg_accuracy=avg_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
        )
    
    def log_performance(self, epoch, batch, batch_loss, batch_accuracy, avg_loss, avg_accuracy, val_loss, val_accuracy):
        self.log_data["epoch"].append(epoch)
        self.log_data["batch"].append(batch)
        self.log_data["batch_loss"].append(batch_loss)
        self.log_data["batch_accuracy"].append(batch_accuracy)
        self.log_data["avg_loss"].append(avg_loss)
        self.log_data["avg_accuracy"].append(avg_accuracy)
        self.log_data["val_loss"].append(val_loss)
        self.log_data["val_accuracy"].append(val_accuracy)

    def save_to_csv(self, file_path):
        """Save the logged data to a CSV file."""
        from pandas import DataFrame
        df = DataFrame(self.log_data)
        df.to_csv(file_path, index=False)
        print(f"Saved performance log to {file_path}")

    def save_to_parquet(self, file_path):
        """Save the logged data to a Parquet file."""
        from pandas import DataFrame
        df = DataFrame(self.log_data)
        df.to_parquet(file_path, index=False)
        print(f"Saved performance log to {file_path}")