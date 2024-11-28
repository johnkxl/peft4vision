from transformers import AutoModelForImageClassification, AutoImageProcessor
from peft import LoraConfig, get_peft_model

def setup_peft_model(base_model_path, label2id, id2label, target_modules):
    model = AutoModelForImageClassification.from_pretrained(
        base_model_path,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"],
    )
    lora_model = get_peft_model(model, peft_config)
    return lora_model


def load_image_processor(preprocessor_path):
    """Load the image processor from the specified path."""
    return AutoImageProcessor.from_pretrained(preprocessor_path)
