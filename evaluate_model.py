from argparse import ArgumentParser
from pathlib import Path
import json
import pickle
import pandas as pd
import logging

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import numpy as np
from tqdm import tqdm
from datasets import Dataset

from src.train_utils import ImageDataset
from download_model import load_siglip_for_image_classification_offline

parser = ArgumentParser(description="Evaluate PEFT-tuned SigLIP model on a test set.")

parser.add_argument('--df', type=Path, required=True, help='Dataset path of test set. Must be a .parquet file with only two columns, "image" and "target".')
parser.add_argument('--peft', type=bool, default=False, help="Evalute the performance on the PEFT-tuned model if `True`.")
parser.add_argument('--batch_size', type=int, default=16, help='(Optional) Batch size. Default 16.')
parser.add_argument('--label2id', type=Path, default=None, help='(Optional) JSON file containing dictionary mapping target class labels to intengers 0 to n_classes - 1.')
parser.add_argument('--out', type=str, required=True, help='JSON file to save evaluation metrics.')

args = parser.parse_args()

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s %(message)s', 
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

DEVICE_TYPE = "cpu"
if torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE_TYPE = "mps"

USE_PEFT = args.peft


def main():
    # Load integer to target class mappings.
    label2id = json.load(open(args.label2id))
    id2label: dict = {i: label for label, i in label2id.items()}
    CLASS_NAMES = [id2label[i] for i in range(len(id2label))]

    # Load PEFT-tuned model and processor
    peft_model, processor = load_siglip_for_image_classification_offline(
        label2id=label2id,
        id2label=id2label,
        peft=USE_PEFT
    )

    # Move the PEFT model to the selected device
    device = torch.device(DEVICE_TYPE)
    peft_model = peft_model.to(device)

    # Load the test dataset and data loader
    test_df = pd.read_parquet(args.df)
    test_ds = Dataset.from_pandas(test_df)
    test_dataset = ImageDataset(test_ds, processor)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Evaluate the model
    logger.info("Evaluating on holdout test set...")
    logger.info(f"Model architecture: {type(peft_model).__name__}")
    logger.info(f"Using device: {device}")

    metrics = evaluate_holdout_set(peft_model, test_loader, device, class_names=CLASS_NAMES)

    # Save metrics
    with open(args.out, 'wb') as outfile:
        pickle.dump(metrics, outfile)
        print(f'Saved evaluation metrics in {args.out}')

    # Print metrics
    for metric, value in metrics.items():
        if isinstance(value, np.ndarray):  # For confusion matrix
            print(f"{metric}:\n{value}")
        else:
            print(f"{metric}: {value}")


def evaluate_holdout_set(model, test_loader, device, class_names=None):
    """
    Evaluate the PEFT-tuned model on a holdout test set.

    Args:
        model: The fine-tuned PEFT model.
        test_loader: DataLoader for the holdout test set.
        device: Device (CPU/CUDA/MPS) to run evaluation.
        class_names: List of class names (for labels).

    Returns:
        metrics_dict: Dictionary containing evaluation metrics.
    """
    model.eval()  # Switch to evaluation mode

    all_labels = []
    all_predictions = []
    all_logits = []

    with torch.no_grad():  # Disable gradient computation for inference
        for batch in tqdm(test_loader, desc="Evaluating on test set"):
            # Load batch to device
            pixel_values, labels = batch
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model.vision_model(pixel_values=pixel_values)
            vision_embeddings = outputs.pooler_output
            logits = model.classifier(vision_embeddings)

            _, predicted = torch.max(logits, 1)  # Predicted class indices

            # Store logits, predictions, and labels
            all_logits.append(logits.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Flatten collected logits, predictions, and labels
    all_logits = np.concatenate(all_logits, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate metrics
    metrics_dict = {}

    # Accuracy
    metrics_dict['accuracy'] = (all_predictions == all_labels).mean()

    # Classification Report (Precision, Recall, F1-Score)
    report = classification_report(all_labels, all_predictions, labels=class_names, target_names=class_names, output_dict=True)
    metrics_dict.update(report)

    # Confusion Matrix
    metrics_dict['confusion_matrix'] = confusion_matrix(all_labels, all_predictions)

    # AUROC (if binary or one-vs-rest for multiclass)
    try:
        metrics_dict['auroc'] = roc_auc_score(all_labels, all_logits, multi_class="ovr")
    except ValueError:
        metrics_dict['auroc'] = "AUROC not applicable for this setup"

    return metrics_dict


if __name__ == "__main__":
    main()