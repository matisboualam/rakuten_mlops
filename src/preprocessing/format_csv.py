import pandas as pd
import os

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

def format_csv(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Required columns for the final CSV
        required_columns = {"image_path", "description", "prdtypecode"}

        # Check and process missing columns
        if "image_path" not in df.columns and {"imageid", "productid"}.issubset(df.columns):
            df["image_path"] = df.apply(lambda row: f"data/raw/img/image_{row['imageid']}_product_{row['productid']}.jpg", axis=1)
        
        if "description" not in df.columns and "designation" in df.columns:
            # Ensure both columns are strings and handle missing values
            df["description"] = df["description"].fillna("") + df["designation"].fillna("")
        
        if "prdtypecode" not in df.columns:
            print("Error: 'prdtypecode' column is missing, cannot proceed.")
            return

        # Map prdtypecode to product labels
        df["prdtypecode"] = df["prdtypecode"].map(product_dict)

        # Retain only the required columns (and the new label column)
        df = df[["image_path", "description", "prdtypecode"]]

        # Save the transformed DataFrame back to the CSV file
        df.to_csv(csv_path, index=False)
        print(f"CSV file saved to {csv_path}")

    except Exception as e:
        print(f"Error occurred while processing the file: {e}")


if __name__ == "__main__":
    csv_path = "data/processed/val_data.csv"
    format_csv(csv_path)
