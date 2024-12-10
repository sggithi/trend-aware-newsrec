import pandas as pd
from collections import Counter

# Re-importing data and processing due to state reset
# Reading the behaviors.tsv file
df = pd.read_csv('behaviors.tsv', sep="\t", names=["Index", "UserID", "Timestamp", "History", "Impression"])

# Splitting items in 'Impression' and collecting all items
all_items = []
for impression in df['Impression'].fillna(''):
    items = impression.split()
    all_items.extend(items)

# Filtering only items ending with '-1'
items_ending_with_minus1 = [item[:-2] for item in all_items if item.endswith('-1')]

# Calculating frequency for items ending with '-1'
item_counts_minus1 = Counter(items_ending_with_minus1)

# Converting to a DataFrame
item_counts_minus1_df = pd.DataFrame(item_counts_minus1.items(), columns=["ItemID", "Frequency"])
item_counts_minus1_df = item_counts_minus1_df.sort_values(by="Frequency", ascending=False)


# Saving the DataFrame to a CSV file
item_counts_minus1_df.to_csv("Impression_frequencies2.csv", index=False)

# Displaying confirmation of saved file
item_counts_minus1_df.head()
