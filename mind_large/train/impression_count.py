# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load item frequencies and behaviors data
item_freq_df = pd.read_csv("Impression_item_frequencies.csv")
df = pd.read_csv('behaviors.tsv', sep="\t", names=["Index", "UserID", "Timestamp", "History", "Impression"])

# Selecting the top 10 items based on frequency that end with "-1"
top_items_ending_1 = item_freq_df[item_freq_df["ItemID"].str.endswith("-1")].head(10)["ItemID"].tolist()

# Converting the 'Timestamp' column to datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Plotting the frequency of each of the top 10 items in Impression over time
plt.figure(figsize=(12, 8))

for item_id in top_items_ending_1:
    # Create a Boolean mask to check if each item_id appears in the Impression column
    df[f'Impression_Contains_{item_id}'] = df['Impression'].apply(lambda x: item_id in x if isinstance(x, str) else False)
    
    # Filter rows where the item is present in Impression and resample in 30-minute intervals
    df_filtered_impression = df[df[f'Impression_Contains_{item_id}']].set_index('Timestamp')
    frequency_impression = df_filtered_impression.resample('30T').size()
    
    # Plotting the frequency for each item in Impression
    plt.plot(frequency_impression.index, frequency_impression.values, marker='o', linestyle='-', label=f"{item_id} in Impression")

# Customize plot
plt.xlabel('Timestamp (30-minute intervals)')
plt.ylabel('Frequency')
plt.title('Frequency of Top 10 Items Ending with -1 in Impression Over Time')
plt.xticks(rotation=45)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05))
plt.tight_layout()

# Save the plot as a PNG file
top_10_items_plot_path = "top_10_items_Impression_frequency.png"
plt.savefig(top_10_items_plot_path)
plt.show()
