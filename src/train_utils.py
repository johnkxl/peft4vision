from transformers import Trainer
import torch
import numpy as np

def create_trainer(model, train_ds, val_ds, args, image_processor, label2id, target, metric):
    def compute_metrics(eval_pred):
        """Compute accuracy on a batch of predictions."""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    def collate_fn(examples):
        pixel_values = torch.stack([example['pixel_values'] for example in examples])
        labels = torch.tensor([label2id[example[target]] for example in examples])
        return {"pixel_values": pixel_values, 'labels': labels}

    return Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
