import pandas as pd
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser(description="Drops target variables that will end up with insufficient samples when passed to `df-analyze`.")

parser.add_argument('--df', type=Path, required=True, help="Dataset path. Must be a .parquet file.")
parser.add_argument('--target', type=str, required=True, help="Target variable for classification.")
parser.add_argument('--out', type=str, required=True, help="File to save cleaned dataset.")
opts = parser.parse_args()

ds_path = opts.df.resolve()
outfile = opts.out
target = opts.target


def main():
    df: pd.DataFrame = pd.read_parquet(ds_path)

    drop_count = 0
    drop_list = []

    for c, count in df[target].value_counts().items():
        split_count = count * 0.1
        df_analyze_split = split_count * 0.4
        if df_analyze_split <= 20:
            print(f'Dropping target class "{c}" due to insufficient samples (n = {count})')
            drop_count += count
            drop_list.append(c)

    df = df[~df[target].isin(drop_list)]
    df.to_parquet(outfile, index=False)

    print("Total dropped:", drop_count)
    print("New sample count is:", df.shape[0])
    print("Target classes remaining:", df[target].nunique())


if __name__ == "__main__":
    main()