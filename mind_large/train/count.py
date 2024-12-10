# Importing necessary libraries
import pandas as pd
import re
from collections import Counter

# Reading the behaviors.tsv file and processing line by line
df =pd.read_csv('behaviors.tsv', sep="\t", names=["Index", "UserID", "Timestamp", "History", "Impression"])

# Splitting all items and calculating their frequencies
all_items =  " ".join(df['Impression'].fillna('')).split()
item_counts = Counter(all_items)

# Converting item counts to a DataFrame
item_counts_df = pd.DataFrame(item_counts.items(), columns=["ItemID", "Frequency"])
item_counts_df = item_counts_df.sort_values(by="Frequency", ascending=False)
# Saving the DataFrame to a CSV file
item_counts_df.to_csv("Impression_item_frequencies.csv", index=False)

# Displaying confirmation of saved file
item_counts_df.head()
