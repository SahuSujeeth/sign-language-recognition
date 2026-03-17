import pandas as pd

data = pd.read_csv("dataset.csv")

print("Before cleaning:", len(data))

# Remove label 6 and 7
data = data[~data['label'].isin(['6', '7'])]

print("After cleaning:", len(data))

data.to_csv("dataset.csv", index=False)

print("6 and 7 removed successfully!")