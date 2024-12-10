# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt


item_freq_df = pd.read_csv("Impression_item_frequencies.csv")

# Selecting the top 100 items based on frequency
top_items = item_freq_df.head(500)["ItemID"].tolist()
#top_items_ending_1 = item_freq_df[item_freq_df["ItemID"].str.endswith("-1")]

# Sorting by frequency and selecting the top 100
#top_10_items_ending_1 = top_items_ending_1.sort_values(by="Frequency", ascending=False).head(10)

# Extracting the ItemIDs as a list
#top_items = top_10_items_ending_1["ItemID"].tolist()

df =pd.read_csv('behaviors.tsv', sep="\t", names=["Index", "UserID", "Timestamp", "History", "Impression"])

# Converting the 'Timestamp' column to datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Plotting the frequency of each of the top 100 items in history
plt.figure(figsize=(12, 8))
count = 0
for item_id in top_items:
    if item_id[-1] == "0":
        continue
    count += 1
    if count == 11:
        break
    item_id_2 = item_id[:-2]
    # Create a Boolean mask to check if each item_id appears in the Impression
    df[f'History_Contains_{item_id_2}'] = df['Impression'].apply(lambda x: item_id_2 in x if isinstance(x, str) else False)
    
    # Filter rows where the item is present in history and resample in 30-minute intervals
    df_filtered_history = df[df[f'History_Contains_{item_id_2}']].set_index('Timestamp')
    frequency_history = df_filtered_history.resample('30T').size()
    
    # Plotting the frequency for each item in history
    plt.plot(frequency_history.index, frequency_history.values, marker='o', linestyle='-', label=f"{item_id} in History")

# Customize plot
plt.xlabel('Timestamp (30-minute intervals)')
plt.ylabel('Frequency')
plt.title('Frequency of Top 10 Items in Impression-1 Over Time in Validation Impression')
plt.xticks(rotation=45)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05))
plt.tight_layout()

# Save the plot as a PNG file
top_100_items_plot_path = "top_10_Impression-1_frequency.png"
plt.savefig(top_100_items_plot_path)
plt.show()
