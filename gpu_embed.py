from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from pandas import DataFrame
import json

import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from src.train_utils import ImageDataset
from download_model import load_siglip_for_image_classification_offline


parser = ArgumentParser(description="Extract embeddings from image dataset.")

parser.add_argument('--df', type=str, required=True, help='Dataset path of test set. Must be a .parquet file with only two columns, "image" and "target".')
parser.add_argument('--use-peft', action='store_true', help='Include this flag to extract embeddings using the PEFT-adapted model.')
parser.add_argument('--out', type=Path, required=True, help='Destination file name (.parquet).')
parser.add_argument('--label2id', type=Path, required=True, help="JSON file mapping target labels to their integer values in the training set.")

args = parser.parse_args()

DEVICE_TYPE = "cpu"
if torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE_TYPE = "mps"


def main():

    USE_PEFT = args.use_peft
    OUTFILE = args.out
    
    label2id: dict = json.load(open(args.label2id))
    id2label: dict = {i:label for label,i in label2id.items()}
    
    model, processor = load_siglip_for_image_classification_offline(label2id, id2label, peft=USE_PEFT)

    ds = Dataset.from_parquet(args.df)
    image_dataset = ImageDataset(ds, processor)
    loader = DataLoader(image_dataset)

    device = torch.device(DEVICE_TYPE)

    df = get_vision_embeddings(model, loader, device)

    df.to_parquet(OUTFILE)
    print(f"Saved image embeddings {OUTFILE}.")


def get_vision_embeddings(model, loader, device):

    model = model.to(device)

    print("Starting image embedding...")
    print(f"Model architechture: {type(model).__name__}")
    print(f"Using device: {device}")
    
    all_embeddings = []
    all_labels = []

    with torch.no_grad():

        for pixel_values, label in tqdm(
            loader,
            total=len(loader),
            desc="Embedding images"
        ):
            pixel_values = pixel_values.to(device)

            ouput = model.vision_model(pixel_values=pixel_values)
            image_embeddings = ouput.pooler_output

            all_embeddings.append(image_embeddings)
            all_labels.append(label.item())

    embeddings = torch.cat(all_embeddings)
    embeddings = embeddings.to('cpu')
    N, p = embeddings.shape
    cols = [f"embed{i:04d}" for i in range(p)]
    df_embed = DataFrame(data=embeddings.numpy(), columns=cols)
    df_embed["target"] = all_labels

    df_embed.columns = df_embed.columns.astype(str)

    return df_embed


if __name__ == "__main__":
    main()