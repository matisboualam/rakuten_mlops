import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(input_csv, test_size, val_size):
    df = pd.read_csv(input_csv)

    assert 0 < test_size < 1, "test_size must be between 0 and 1"
    assert 0 < val_size < 1, "val_size must be between 0 and 1"
    # assert unseen_size + test_size + val_size < 1, "Sum of unseen, test, and val sizes must be less than 1"

    train_df, temp_df = train_test_split(df, test_size=(test_size + val_size), random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=test_size / (test_size + val_size), random_state=42)

    train_df.to_csv("data/processed/train.csv", index=False)
    val_df.to_csv("data/processed/val.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size", type=float, required=True)
    parser.add_argument("--val_size", type=float, required=True)

    args = parser.parse_args()

    split_data("data/raw/merged.csv", args.test_size, args.val_size)
