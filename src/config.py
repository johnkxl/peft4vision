from pathlib import Path

ROOT = Path(__file__).parent.parent
SIGLIP_PATH = ROOT / "downloaded_models/siglip_so400m_patch14_384"

CONFIG = {
    "model_checkpoint": "google/siglip-so400m-patch14-384",
    "siglip_model": SIGLIP_PATH / "model",
    "siglip_preprocessor": SIGLIP_PATH / "preprocessor",
    "siglip_peft_adapter": SIGLIP_PATH / "peft_adapter",
    
    "train_args": {
        "output_dir": "finetuned-lora-model",  # Directory to save model
        "remove_unused_columns": False,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "learning_rate": 3e-2,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "per_device_eval_batch_size": 2,
        "num_train_epochs": 5,
        "logging_steps": 10,
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "label_names": ["target"]
    },
}