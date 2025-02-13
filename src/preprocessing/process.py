import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

# Dictionary mapping prdtypecode to labels
product_dict = {
    2583: "pool accessories",
    1560: "home furnishings and decoration",
    1300: "model making",
    2060: "home accessories and decorations",
    2522: "stationery",
    1280: "children's toys",
    2403: "literature series",
    2280: "non-fiction books",
    1920: "textile accessories and decorations",
    1160: "trading card games",
    1320: "nursery products",
    10: "foreign literature",
    2705: "historical literature",
    1140: "figurines",
    2582: "garden accessories and decorations",
    40: "video games",
    2585: "gardening accessories and tools",
    1302: "outdoor accessories",
    1281: "children's games",
    50: "gaming accessories",
    2462: "sets of gaming or video game accessories",
    2905: "downloadable video games",
    60: "video game consoles",
    2220: "pet accessories",
    1301: "articles for newborns and babies",
    1940: "food",
    1180: "figurines to paint and assemble"
}

def format_csv(x_csv, y_csv, unseen_size):
    if not os.path.exists(x_csv):
        print(f"Error: File not found at {x_csv}")
        return
    if not os.path.exists(y_csv):
        print(f"Error: File not found at {y_csv}")
        return
    else:
        x_data = pd.read_csv(x_csv)
        y_data = pd.read_csv(y_csv)
        assert len(x_data) == len(y_data)
        df = pd.concat([x_data, y_data], axis=1)

    new_df = pd.DataFrame()    
    new_df["image_path"] = df.apply(lambda row: f"/workspace/data/raw/img/image_{row['imageid']}_product_{row['productid']}.jpg", axis=1)
    new_df["description"] = df["designation"].fillna("") + " " + df["description"].fillna("")
    new_df["prdtypecode"] = df["prdtypecode"].map(product_dict)
    new_df.to_csv('data/raw/merged.csv', index=False)
    print(f"✅ Données fusionnées et sauvegardées dans 'data/raw/merged.csv'")

    data_df, unseen_df = train_test_split(new_df, test_size=unseen_size, random_state=42)
    data_df.to_csv('data/processed/data.csv', index=False)
    unseen_df.to_csv('data/processed/unseen.csv', index=False)
    print(f"✅ Données divisées en deux groupes 'data' et 'unseen' dans le dossier data/processed")
    print(f"Number of lines in 'data.csv': {len(data_df)}")
    print(f"Number of lines in 'unseen.csv': {len(unseen_df)}")

if __name__ == "__main__":
    x_data_path = '/workspace/data/raw/x_data.csv'
    y_data_path = '/workspace/data/raw/y_data.csv'
    parser = argparse.ArgumentParser()
    parser.add_argument("--unseen_size", type=float, required=True)

    args = parser.parse_args()

    format_csv(x_data_path, y_data_path, args.unseen_size)
