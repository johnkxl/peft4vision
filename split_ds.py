from sklearn.model_selection import train_test_split
from pathlib import Path
from pandas import DataFrame, read_parquet
from argparse import ArgumentParser, RawTextHelpFormatter
import json

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


def split_dataset(df: DataFrame, target: str, train_size: float, grouper: str = None) -> tuple[DataFrame, DataFrame]:
    """
    Split a dataset into training and validation sets, with optional grouping by a specific column.

    Args:
        df (DataFrame): The input dataset.
        target (str): The column name representing the target variable for stratification.
        train_size (float): Proportion of the data to use for training.
        grouper (str, optional): Column name to group by for splitting. Defaults to None.

    Returns:
        tuple[DataFrame, DataFrame]: Training and validation DataFrames.
    """
    X_train: DataFrame
    X_val: DataFrame

    if grouper:
        # Create datafram with one row per unique leaf.
        groups = df.groupby(grouper).first().reset_index()
        # Perform split on the group-level data.
        train_ids, val_ids = train_test_split(
            groups[grouper],
            test_size=1-train_size,
            stratify=groups[target],  # Stratify based on target
            random_state=42           # For reproducibility
        )
        # Create trrain and test splits by IDs assigned to each.
        X_train = df[df[grouper].isin(train_ids)]
        X_val = df[df[grouper].isin(val_ids)]

    else:
        # Standard split
        X_train, X_val = train_test_split(
            df,
            test_size=1-train_size,
            stratify=df[target],  # Stratify based on target
            random_state=42       # For reproducibility
        )

    print("Training set size:", X_train.shape)
    print("Validation set size:", X_val.shape)

    return X_train, X_val


if __name__ == "__main__":
    main()