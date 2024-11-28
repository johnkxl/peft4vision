from df_analyze.embedding.embed import get_embeddings
from df_analyze.embedding.datasets import VisionDataset
from download_model import load_siglip_offline

import torch

from argparse import ArgumentParser
from pathlib import Path
import os

parser = ArgumentParser(description="Extract image embeddings")

parser.add_argument('--df', type=Path, required=True, help='Dataset path of test set.')
parser.add_argument('--target', type=str, required=True, help='Target for classifiction.')
parser.add_argument('--out', type=Path, required=True, help='Destination file name (.parquet).')

args = parser.parse_args()

DF_PATH = args.df
OUTFILE = args.out
TARGET = args.target


def main() -> None:
    
    if not torch.cuda.is_available():
        # Limit multiprocessing
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

        # Set PyTorch DataLoader workers to 0 globally (no multiprocessing)
        os.environ["NUM_WORKERS"] = "0"

    # Use df-embed to extract image embeddings
    ds = VisionDataset(datapath=DF_PATH, name=None)
    model, processor = load_siglip_offline(peft=False)
    
    df = get_embeddings(
        ds=ds,  # type: ignore
        processor=processor,  # type: ignore
        model=model,  # type: ignore
        batch_size=2,
        load_limit=None
    )

    # print(df)
    df.to_parquet(OUTFILE)
    print(f"Saved embeddings to {OUTFILE}")

    return


if __name__ == "__main__":
    main()