import pandas as pd

# Load dataset WITHOUT header
data = pd.read_csv("dataset.csv", header=None)

# Create correct column names
columns = [f"f{i}" for i in range(63)]
columns.append("label")

data.columns = columns

# Save fixed dataset
data.to_csv("dataset_fixed.csv", index=False)

print("Dataset header fixed successfully!")