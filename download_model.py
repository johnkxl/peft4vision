from pathlib import Path
from typing import cast

from transformers import AutoModel, AutoProcessor, AutoModelForImageClassification, AutoImageProcessor
from transformers.models.siglip.modeling_siglip import SiglipModel
from transformers.models.siglip.processing_siglip import SiglipProcessor
from peft import PeftModel


ROOT = Path(__file__).parent
SIGLIP_PATH = ROOT / "downloaded_models/siglip_so400m_patch14_384/"

SIGLIP_PEFT_ADAPTER = SIGLIP_PATH / "peft_adapter"

SIGLIP_MODEL = SIGLIP_PATH / "model"
SIGLIP_MODEL_FILES = [
    SIGLIP_MODEL / "config.json",
    SIGLIP_MODEL / "model.safetensors",
]

SIGLIP_PREPROCESSOR = ROOT / SIGLIP_PATH / "preprocessor"
SIGLIP_PREPROCESSOR_FILES = [
    SIGLIP_PREPROCESSOR / "preprocessor_config.json",
    SIGLIP_PREPROCESSOR / "special_tokens_map.json",
    SIGLIP_PREPROCESSOR / "spiece.model",
    SIGLIP_PREPROCESSOR / "tokenizer_config.json",
]


def download_siglip_model() -> None:
    model = cast(
        SiglipModel, AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
    )
    model.save_pretrained(SIGLIP_MODEL)
    print(f"Saved model to {SIGLIP_MODEL}")

    processor = cast(
        SiglipProcessor, AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    )
    processor.save_pretrained(SIGLIP_PREPROCESSOR)
    print(f"Saved preprocessor to {SIGLIP_PREPROCESSOR}")


def load_siglip_offline(peft=False) -> tuple[SiglipModel, SiglipProcessor]:
    """Returns model and tokenizer. The model is the peft trained model, `peft` set to `True`."""
    model = AutoModel.from_pretrained(SIGLIP_MODEL, local_files_only=True)

    tokenizer = cast(
        SiglipProcessor,
        AutoProcessor.from_pretrained(SIGLIP_PREPROCESSOR, local_files_only=True),
    )

    if peft:
        # Wrap the model with the PEFT adapter.
        model = PeftModel.from_pretrained(model, SIGLIP_PEFT_ADAPTER)
    
    model = cast(SiglipModel, model)

    return model, tokenizer


def load_siglip_for_image_classification_offline(
    label2id: dict,
    id2label: dict,
    peft=False
) -> tuple[AutoModelForImageClassification, AutoImageProcessor]:
    model = AutoModelForImageClassification.from_pretrained(
        SIGLIP_MODEL,
        local_files_only=True,
        label2id=label2id,
        id2label=id2label
    )
    processor = AutoImageProcessor.from_pretrained(SIGLIP_PREPROCESSOR, local_files_only=True)

    if peft:
        # Wrap the model with the PEFT adapter.
        model = PeftModel.from_pretrained(model, SIGLIP_PEFT_ADAPTER)
    
    return model, processor


if __name__ == "__main__":
    download_siglip_model()