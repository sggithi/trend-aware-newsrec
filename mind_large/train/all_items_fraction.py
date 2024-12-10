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

# Plotting the frequency ratio of each of the top items in history
plt.figure(figsize=(12, 8))
count = 0

for item_id in top_items:
    # Select items ending in `-1` and find corresponding `-0` versions
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
    
    # Avoid division by zero and calculate the frequency ratio `-1 / -0`
    frequency_ratio = frequency_1 / frequency_0.replace(0, 0.5)  # Replace 0 with NA to avoid division by zero

    # Plotting the frequency ratio for each item in history
    plt.plot(frequency_ratio.index, frequency_ratio.values, marker='o', linestyle='-', label=f"{base_item_id} (-1 / -0 Ratio)")

# Customize plot
plt.xlabel('Timestamp (30-minute intervals)')
plt.ylabel('Frequency Ratio (-1 / -0)')
plt.title('Frequency Ratio of Top 10 Items (Impression-1 / Impression-0) Over Time in Train Set')
plt.xticks(rotation=45)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05))
plt.tight_layout()

# Save the plot as a PNG file
top_10_items_ratio_plot_path = "top_10_items_Impression_1_0_ratio.png"
plt.savefig(top_10_items_ratio_plot_path)
plt.show()
