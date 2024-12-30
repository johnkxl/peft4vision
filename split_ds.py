import json
import numpy as np
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
parser.add_argument('--grouper', type=str, default=None, help=GROUP_HELP_STR)
parser.add_argument('--outdir', required=True, type=Path, help='Directory to save train_valid and test splits.')

args = parser.parse_args()

DF_PATH: Path = args.df
TARGET: str = args.target
TRAIN_SIZE: float = args.train_size
GROUPER: str = args.grouper
OUTDIR: Path = args.outdir


def main():

    OUTDIR.mkdir(parents=True, exist_ok=True)

    df: DataFrame = read_parquet(DF_PATH)

    # Create feature subset for use in training and embedding.
    feature_subset = ['image', TARGET]
    
    # Map labels to [0, c-1], c:=number of target classes to match embeddeding format requirements
    label2id = create_label2id_dict(df, TARGET)
    df[TARGET] = df[TARGET].map(label2id)
    with open(OUTDIR / "label2id.json", 'w') as outfile:
        json.dump(label2id, outfile)
        print(f'Saved mappings for target "{TARGET} as label2id.json')

    train_valid, test = split_dataset(df, TARGET, TRAIN_SIZE, 'ID' if GROUPER else None)


    # Save splits with complete featureset.
    train_valid.to_parquet(OUTDIR / "train.parquet", index=False)
    test.to_parquet(OUTDIR / "test.parquet", index=False)

    print(f"Saved {100 * TRAIN_SIZE:.2f}% to train.parquet")
    print(f"Saved {100 * (1 - TRAIN_SIZE):.2f}% to test.parquet")


    # Save splits with only image and target columns
    test = test[feature_subset]
    test.rename(columns={TARGET: "target"}, inplace=True)
    test.to_parquet(OUTDIR / "test_image_target.parquet", index=False)

    # Training split has optional grouper column.
    if GROUPER:
        df.rename(columns={GROUPER: 'ID'}, inplace=True)
        feature_subset.insert(0, 'ID')

    train_valid = train_valid[feature_subset]
    train_valid.rename(columns={TARGET: "target"}, inplace=True)
    train_valid.to_parquet(OUTDIR / "train_image_target.parquet", index=False)
    
    print(f"Saved {100 * TRAIN_SIZE:.2f}% to train_image_target.parquet")
    print(f"Saved {100 * (1 - TRAIN_SIZE):.2f}% to test_image_target.parquet")

    return


def create_label2id_dict(df: DataFrame, target: str) -> dict[str|int, int]:
    """
    Create dictionary mapping of target labels to ids.

    Args:
        df (DataFrame): The input DataFrame.
        target (str): The name of the target variable column.
    
    Returns:
        dict[str|int, int]: dictionary mapping keys of type str or int to int IDs.
    """
    if df[target].dtype == np.int64:
        return {int(label): i for i, label in enumerate(df[TARGET].unique())}
    
    return {label: i for i, label in enumerate(df[TARGET].unique())}


if __name__ == "__main__":
    main()