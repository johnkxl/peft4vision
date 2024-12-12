from argparse import ArgumentParser
from pathlib import Path
import logging

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from peft import LoraConfig, get_peft_model
from datasets import load_dataset

from src.train_utils import ImageDataset, EarlyStopping, evaluate, print_trainable_parameters
from download_model import load_siglip_offline, SIGLIP_PEFT_ADAPTER


parser = ArgumentParser(description='Fine-tune PEFT adapter.')
parser.add_argument('--train_ds', type=Path, required=True, help='Path to training dataset.')
parser.add_argument('--test_size', type=float, default=0.111, help='Validation split size.')
parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to run. Set to 5 by default.')
parser.add_argument('--learn_rate', type=float, default=5e-5, help='Learning rate. Set to 5e-5 by default.')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size. Default 16.')
parser.add_argument('--log_interval', type=int, default=10, help='Performance reporting interval. Report on training performance every `log_interval` batches. Default 10.')
args = parser.parse_args()

DS_PATH = args.train_ds.resolve()
TEST_SIZE = args.test_size
# NUM_EPOCHS = args.num_epochs
LEARN_RATE = args.learn_rate

DEVICE_TYPE = "cpu"
if torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE_TYPE = "mps"

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def main():

    # Load dataset into training and validation splits
    dataset = load_dataset('parquet', data_files=[str(DS_PATH)])
    splits = dataset['train'].train_test_split(test_size=TEST_SIZE)
    train_ds, valid_ds = splits['train'], splits['test']

    # Load base model and processor
    base_model, processor = load_siglip_offline()

    # Freeze most of the model's parameters
    for param in base_model.parameters():
        param.requires_grad = False

    # Define a PEFT configuration using LoRA
    peft_config = LoraConfig(
        task_type="IMAGE_CLASSIFICATION",  # Define the task type (delete this is causing errors)
        inference_mode=False,              # Enable training
        r=16,                              # Low-rank dimension
        lora_alpha=32,                     # Scaling factor
        lora_dropout=0.1,                  # Dropout
        target_modules=[
            # "k_proj",
            "v_proj",
            "q_proj",
            # "out_proj",
        ]
    )

    # Wrap the base model with the PEFT model
    peft_model = get_peft_model(base_model, peft_config)
    print_trainable_parameters(peft_model)

    # Move the PEFT model to the selected device
    device = torch.device(DEVICE_TYPE)
    peft_model = peft_model.to(device)

    # Wrap datasets
    train_dataset = ImageDataset(train_ds, processor)
    valid_dataset = ImageDataset(valid_ds, processor)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    # Training utils
    criterion = nn.CrossEntropyLoss()  # Loss function
    optimizer = Adam(peft_model.parameters(), lr=LEARN_RATE)  # Updates parameters

    scheduler = ReduceLROnPlateau(
        optimizer,                  # Optimizer instance
        mode='min',                 # Minimizing validation loss
        factor=0.1,                 # Multiply LR by this factor on plateau
        patience=3,                 # Number of epochs to wait before reducing LR
        verbose=True                # Print LR reduction messages
    )

    NUM_EPOCHS = args.num_epochs    # Maximum number of epochs
    patience = 5                    # Patience for early stopping
    delta = 0.001                   # Minimum improvement for early stopping

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True)

    # Log the start of training
    logger.info("Training started.")
    logger.info(f"Model architecture: {type(base_model).__name__}")
    logger.info(f"Using device: {device}")
    log_interval = args.log_interval

    for epoch in range(NUM_EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        peft_model.train()  # Training mode
        running_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch_idx, (pixel_values, labels) in tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
            unit="batch"
        ):
            # Move data to the device (GPU/CPU)
            pixel_values, labels = pixel_values.to(device), labels.to(device)

            optimizer.zero_grad()  # Clear gradients from the previous step

            # Forward pass
            outputs = peft_model.vision_model(pixel_values=pixel_values)
            logits = outputs.pooler_output
            loss = criterion(logits, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

            # Log batch performance every few batches
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = running_loss / (batch_idx + 1)
                accuracy = (correct_predictions / total_predictions) * 100
                logger.info(
                    f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}"
                    f" - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
                )

        # Evaluate on validation set
        val_loss, val_accuracy = evaluate(peft_model, valid_loader, criterion, device)
        logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Training Loss: {running_loss / len(train_loader):.4f}, ")
        logger.info(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        # Step the scheduler
        scheduler.step(val_loss)

        # Check for early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

    # Save the model after training finishes
    print("Training complete. Saving the model...")
    peft_model.save_pretrained(SIGLIP_PEFT_ADAPTER)
    print(f"PEFT-tuned model saved to: {SIGLIP_PEFT_ADAPTER}")