from pathlib import Path
from pandas import DataFrame

from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split


def split_dataset(
        
        df: DataFrame, target: str, train_size: float, grouper: str = None
    
    ) -> tuple[DataFrame, DataFrame]:
    """
    Split a DataFrame into training and validation sets, with optional grouping by a specific column.

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
        # Create df with ID group representatives
        grouped_df = df.groupby(grouper).first().reset_index()
        # Perform train-test split if the repensentatives.
        train_ids, valid_ids = train_test_split(
            grouped_df[grouper],
            train_size=train_size,
            stratify=grouped_df[target],
            random_state=42
        )
        # Blow up group representatives to whole groups.
        X_train = df[df[grouper].isin(train_ids)]
        X_val = df[df[grouper].isin(valid_ids)]

    else:
        # Standard split
        X_train, X_val = train_test_split(
            df,
            train_size=train_size,
            stratify=df[target],    # Stratify based on target
            random_state=42         # For reproducibility
        )

    print("Training set size:", X_train.shape)
    print("Validation set size:", X_val.shape)

    return X_train, X_val


def load_dataset_splits(
        
        ds_path: Path, target: str, test_size: float, grouper: str = None
    
    ) -> tuple[Dataset, Dataset]:
    """
    Split a dataset into training and validation sets, with optional grouping by a specific column.

    Args:
        ds_path (Path): The input dataset parquet file path.
        target (str): The column name representing the target variable for stratification.
        test_size (float): Proportion of the data to use for testing.
        grouper (str, optional): Column name to group by for splitting. Defaults to None.

    Returns:
        tuple[Dataset, Dataset]: Training and validation Datasets.
    """
    dataset: DatasetDict = load_dataset('parquet', data_files=[str(ds_path)])

    train_ds: Dataset
    valid_ds: Dataset

    df = dataset['train'].to_pandas()

    train_df, valid_df = split_dataset(
        df=df,
        target=target,
        train_size=1-test_size,
        grouper=grouper
    )
    # Transfer DataFrames to Datasets
    train_ds = Dataset.from_pandas(train_df)
    valid_ds = Dataset.from_pandas(valid_df)
    
    return train_ds, valid_ds