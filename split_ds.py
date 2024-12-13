import json
from pathlib import Path
from pandas import DataFrame, read_parquet
from argparse import ArgumentParser, RawTextHelpFormatter

from src.dataset import split_dataset

parser = ArgumentParser(
    description="Randomly split dataset into train and test sets with similar target distributions",
    formatter_class=RawTextHelpFormatter,
)

parser.add_argument('--df', type=Path, required=True, help='.parquet file of entire image dataset.')
parser.add_argument('--target', type=str, required=True, help='Classification target variable.')
parser.add_argument('--train_size', type=float, required=True, help='Percentage of dataset to use for training.')
GROUP_HELP_STR = """
The (string) name of the grouping variable (if one is present) which will be
used to ensure samples within the same group do not end up in both train and
validation splits.

"""
parser.add_argument('--groupby', type=str, default=None, help=GROUP_HELP_STR)
parser.add_argument('--outdir', required=True, type=Path, help='Directory to save train_valid and test splits.')

args = parser.parse_args()

DF_PATH: Path = args.df
TARGET: str = args.target
TRAIN_SIZE: float = args.train_size
OUTDIR: Path = args.outdir


def main():

    OUTDIR.mkdir(parents=True, exist_ok=True)

    df: DataFrame = read_parquet(DF_PATH)

    # Map labels to [0, c-1], c:=number of target classes to match embeddeding format requirements
    label2id = {label: i for i, label in enumerate(df[TARGET].unique())}
    df[TARGET] = df[TARGET].map(label2id)
    with open(OUTDIR / "label2id.json", 'w') as outfile:
        json.dump(label2id, outfile)
        print(f'Saved mappings for target "{TARGET} as label2id.json')

    train_valid, test = split_dataset(df, TARGET, TRAIN_SIZE, args.groupby)

    # Save splits with complete featureset.
    train_valid.to_parquet(OUTDIR / "train.parquet", index=False)
    test.to_parquet(OUTDIR / "test.parquet", index=False)

    print(f"Saved {100 * TRAIN_SIZE:.2f}% to train.parquet")
    print(f"Saved {100 * (1 - TRAIN_SIZE):.2f}% to test.parquet")

    # Save splits with only image and target columns
    train_valid = train_valid[['image', TARGET]]
    train_valid.rename(columns={TARGET: "target"}, inplace=True)
    train_valid.to_parquet(OUTDIR / "train_image_target.parquet", index=False)

    test = test[['image', TARGET]]
    test.rename(columns={TARGET: "target"}, inplace=True)
    test.to_parquet(OUTDIR / "test_image_target.parquet", index=False)

    print(f"Saved {100 * TRAIN_SIZE:.2f}% to train_image_target.parquet")
    print(f"Saved {100 * (1 - TRAIN_SIZE):.2f}% to test_image_target.parquet")

    return


if __name__ == "__main__":
    main()