from argparse import ArgumentParser
from pathlib import Path

from src.preprocess import load_and_split_dataset, preprocess_images, get_transforms
from src.model_utils import setup_peft_model, load_image_processor
from src.train_utils import create_trainer
from src.config import CONFIG

from transformers import TrainingArguments
import evaluate
import torch


def main():

    parser = ArgumentParser(description='Fine-tune PEFT adapter.')
    parser.add_argument('--train_ds', type=Path, required=True, help='Path to training dataset.')
    parser.add_argument('--test_size', type=float, default=0.111, help='Validation split size.')
    parser.add_argument('--use_fp16', action='store_true', help='Enable FP16 training. Requires a GPU.')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to run. Set to 5 by default.')
    parser.add_argument('--learn_rate', type=float, default=3e-2, help='Learning rate. Set to 3e-2 by default.')
    args = parser.parse_args()

    # Check for GPU and configure FP16
    use_fp16 = args.use_fp16 and torch.cuda.is_available()
    if args.use_fp16:
        if torch.cuda.is_available():
            print("FP16 training is enabled as GPU is available.")
        else:
            print("Warning: FP16 training requires a GPU but none is available. Training will proceed in FP32.")
    else:
        print("FP16 training is disabled.")

    # Load dataset and preprocess
    DS_PATH = args.train_ds.resolve()
    train_ds, val_ds, label2id, id2label = load_and_split_dataset(str(DS_PATH), "target", args.test_size)

    # Load image processor and transforms
    image_processor = load_image_processor(CONFIG["siglip_preprocessor"])
    train_transforms, val_transforms = get_transforms(image_processor)

    train_ds.set_transform(lambda batch: preprocess_images(batch, train_transforms))
    val_ds.set_transform(lambda batch: preprocess_images(batch, val_transforms))

    # Setup PEFT model
    lora_model = setup_peft_model(
        CONFIG["siglip_model"],
        label2id,
        id2label,
        ["k_proj", "v_proj", "q_proj", "out_proj"]
    )

    # Prepare training arguments
    train_args = TrainingArguments(
        output_dir="finetuned-lora-model",  # Directory to save model
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learn_rate,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=8,
        num_train_epochs=args.num_epochs,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        label_names=["labels"],
        fp16=use_fp16,
    )

    # Define evaluation metric
    metric = evaluate.load("accuracy")

    # Create trainer
    trainer = create_trainer(
        lora_model,
        train_ds,
        val_ds,
        train_args,
        image_processor,
        label2id,
        "target",
        metric
    )

    # Train and evaluate
    trainer.train()
    print("Evaluate PEFT-adapted mode:", trainer.evaluate())

    # Save PEFT adapter
    lora_model.save_pretrained(CONFIG["siglip_peft_adapter"])
