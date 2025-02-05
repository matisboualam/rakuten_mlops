import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(input_path, output_dir, test_size=0.2, val_size=0.1, unseen_size=0.1):
    df = pd.read_csv(input_path)

    unseen_df = df.sample(frac=unseen_size, random_state=42)
    df = df.drop(unseen_df.index)

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=val_size / (1 - test_size), random_state=42)

    unseen_df.to_csv(f"{output_dir}/unseen.csv", index=False)
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/val.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)

    print(f"✅ Split terminé et sauvegardé dans {output_dir}")

if __name__ == "__main__":
    split_data("data/processed/merged.csv", "data/splits")
