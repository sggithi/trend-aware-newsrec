import pandas as pd
from collections import Counter

# Reading the behaviors.tsv file
df = pd.read_csv('behaviors.tsv', sep="\t", names=["Index", "UserID", "Timestamp", "History", "Impression"])

# Splitting items in 'Impression' and collecting all items
all_items = []
for impression in df['Impression'].fillna(''):
    items = impression.split()
    all_items.extend(items)

# Calculating frequency of each item including -0 and -1
item_counts = Counter(all_items)

# Converting item counts to a DataFrame
item_counts_df = pd.DataFrame(item_counts.items(), columns=["ItemID", "Frequency"])
item_counts_df = item_counts_df.sort_values(by="Frequency", ascending=False)

# Saving the DataFrame to a CSV file
item_counts_df.to_csv("Impression_item_frequencies_separated.csv", index=False)

# Displaying confirmation of saved file
item_counts_df.head()
