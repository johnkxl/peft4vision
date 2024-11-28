from PIL import Image
from io import BytesIO

from datasets import load_dataset, Dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor
)


def load_and_split_dataset(path, target: str, test_size=0.1) -> tuple[Dataset, Dataset, dict[str,int], dict[int,str]]:
    dataset = load_dataset("parquet", data_files=[path])
    
    labels: set[str] = set(dataset['train'][target])
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    
    splits = dataset['train'].train_test_split(test_size=test_size)

    return splits['train'], splits['test'], label2id, id2label


def preprocess_images(example_batch, transforms):
    """Apply transformations across a batch."""
    example_batch['pixel_values'] = [
        transforms(Image.open(BytesIO(image)).convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch


def get_transforms(image_processor):
    """Generate training and validation transforms based on the image processor."""
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

    train_transforms = Compose(
        [
            RandomResizedCrop(image_processor.size['height']),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

    val_transforms = Compose(
        [
            Resize(image_processor.size['height']),
            CenterCrop(image_processor.size['height']),
            ToTensor(),
            normalize,
        ]
    )
    return train_transforms, val_transforms
