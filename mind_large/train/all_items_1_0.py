# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
item_freq_df = pd.read_csv("Impression_item_frequencies.csv")

# Selecting the top 900 items based on frequency
top_items = item_freq_df.head(900)["ItemID"].tolist()

# Load the behaviors dataset
df = pd.read_csv('behaviors.tsv', sep="\t", names=["Index", "UserID", "Timestamp", "History", "Impression"])

# Convert the 'Timestamp' column to datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Plotting the frequency of each of the top items in history
plt.figure(figsize=(12, 8))
count = 0

for item_id in top_items:
    # Separate `ItemID` ending with `-1` and `-0`
    if item_id[-1] != "1":
        continue
    
    # Extract the base `ItemID` without the suffix
    base_item_id = item_id[:-2]
    count += 1
    if count == 11:
        break
    
    # Create boolean masks to check if each `ItemID` with `-1` or `-0` appears in the Impression
    df[f'Impression_Contains_{base_item_id}_1'] = df['Impression'].apply(lambda x: f"{base_item_id}-1" in x if isinstance(x, str) else False)
    df[f'Impression_Contains_{base_item_id}_0'] = df['Impression'].apply(lambda x: f"{base_item_id}-0" in x if isinstance(x, str) else False)
    
    # Filter rows where the `ItemID` with `-1` or `-0` is present and resample in 30-minute intervals
    df_filtered_1 = df[df[f'Impression_Contains_{base_item_id}_1']].set_index('Timestamp')
    df_filtered_0 = df[df[f'Impression_Contains_{base_item_id}_0']].set_index('Timestamp')
    
    frequency_1 = df_filtered_1.resample('30T').size()
    frequency_0 = df_filtered_0.resample('30T').size()
    
    # Plotting the frequency for each item in history for both `-1` and `-0`
    plt.plot(frequency_1.index, frequency_1.values, marker='o', linestyle='-', label=f"{base_item_id}-1 Frequency")
    plt.plot(frequency_0.index, frequency_0.values, marker='x', linestyle='--', label=f"{base_item_id}-0 Frequency")

# Customize plot
plt.xlabel('Timestamp (30-minute intervals)')
plt.ylabel('Frequency')
plt.title('Frequency of Top 10 Items (Impression-1 / Impression-0) Over Time in Train Set')
plt.xticks(rotation=45)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05))
plt.tight_layout()

# Save the plot as a PNG file
top_10_items_plot_path = "top_10_items_Impression_1_0_frequency.png"
plt.savefig(top_10_items_plot_path)
plt.show()
