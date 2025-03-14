import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

def split_data(input_csv, test_size, val_size):
    df = pd.read_csv(input_csv)

    assert 0 < test_size < 1, "test_size must be between 0 and 1"
    assert 0 < val_size < 1, "val_size must be between 0 and 1"
    # assert unseen_size + test_size + val_size < 1, "Sum of unseen, test, and val sizes must be less than 1"

    train_df, temp_df = train_test_split(df, test_size=(test_size + val_size), random_state=42, stratify=df["prdtypecode"])
    val_df, test_df = train_test_split(temp_df, test_size=test_size / (test_size + val_size), random_state=42, stratify=temp_df["prdtypecode"])

    train_df.to_csv("data/processed/train.csv", index=False)
    val_df.to_csv("data/processed/val.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    print(f"Number of rows in train.csv: {len(train_df)}")
    print(f"Number of rows in val.csv: {len(val_df)}")
    print(f"Number of rows in test.csv: {len(test_df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--val_size", type=float, default=0.2)

    args = parser.parse_args()

    split_data("data/processed/data.csv", args.test_size, args.val_size)
