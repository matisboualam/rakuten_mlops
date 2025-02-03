import pandas as pd
from format_csv import format_csv
from sklearn.model_selection import train_test_split

# Load your data from CSV files
x_data = pd.read_csv('data/raw/x_data.csv')  # Replace with the path to your x_data.csv
y_data = pd.read_csv('data/raw/y_data.csv')  # Replace with the path to your y_data.csv

# Ensure the x_data and y_data are aligned correctly (i.e., rows correspond)
assert len(x_data) == len(y_data), "x_data and y_data must have the same number of rows."

# Combine x_data and y_data into one DataFrame (to ensure alignment)
data = pd.concat([x_data, y_data], axis=1)

# Split the combined data into Unseen Data (70%) and Train Data (30%)
train_data, unseen_data = train_test_split(data, test_size=0.7, random_state=42)

# Now split the train_data into training (80%) and validation (20%)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# format the data and save the splits to new CSV files
train_data = format_csv(df=train_data)
val_data = format_csv(df=val_data)
unseen_data = format_csv(df=unseen_data)

# Save the resulting splits to new CSV files
train_data.to_csv('data/processed/train_data.csv', index=False)
val_data.to_csv('data/processed/val_data.csv', index=False)
unseen_data.to_csv('data/processed/unseen_data.csv', index=False)

print(f"Training data saved to 'data/processed/train_data.csv'.")
print(f"Validation data saved to 'data/processed/val_data.csv'.")
print(f"Unseen data saved to 'data/processed/unseen_data.csv'.")
