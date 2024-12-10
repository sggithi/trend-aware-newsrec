# Importing necessary libraries
import pandas as pd
import re
from collections import Counter

# Reading the behaviors.tsv file and processing line by line
data = []
with open("behaviors.tsv", "r") as file:
    for line in file:
        # Matching and extracting fields from each line
        match = re.match(r"(\S+)\s+(\S+)\t(.+)", line.strip())
        if match:
            index = match.group(1)
            user_id = match.group(2)
            timestamp_and_items = match.group(3)

            # Extracting timestamp and items
            timestamp_match = re.match(r"(\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2} (?:AM|PM))\s+(.+)", timestamp_and_items)
            if timestamp_match:
                timestamp = timestamp_match.group(1)
                items = timestamp_match.group(2)
            else:
                timestamp = None
                items = timestamp_and_items

            # Appending processed data to the list
            data.append([index, user_id, timestamp, items])

# Converting the list to a DataFrame
df = pd.DataFrame(data, columns=["Index", "UserID", "timestamp", "items"])

# Splitting all items and calculating their frequencies
all_items = " ".join(df['items']).split()
item_counts = Counter(all_items)

# Converting item counts to a DataFrame
item_counts_df = pd.DataFrame(item_counts.items(), columns=["ItemID", "Frequency"])
item_counts_df = item_counts_df.sort_values(by="Frequency", ascending=False)
# Saving the DataFrame to a CSV file
item_counts_df.to_csv("item_frequencies.csv", index=False)

# Displaying confirmation of saved file
item_counts_df.head()
