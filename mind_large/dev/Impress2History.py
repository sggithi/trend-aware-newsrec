# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt



top_items = ['N31958', 'N36779', 'N6916', "N20036", 'N53572', 'N5472', 'N29862', 'N49285', 'N42844', "N55237"]

df =pd.read_csv('behaviors.tsv', sep="\t", names=["Index", "UserID", "Timestamp", "History", "Impression"])

# Converting the 'Timestamp' column to datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
#print(df['History'])
# Plotting the frequency of each of the top 100 items in history
plt.figure(figsize=(12, 8))
count = 0
for item_id in top_items:
    if item_id[-1] == "0":
        continue
    count += 1
    if count == 11:
        break
    item_id_2 = item_id
    # Create a Boolean mask to check if each item_id appears in the Impression
    df[f'History_Contains_{item_id_2}'] = df['History'].apply(lambda x: item_id_2 in x if isinstance(x, str) else False)
    
    # Filter rows where the item is present in history and resample in 30-minute intervals
    df_filtered_history = df[df[f'History_Contains_{item_id_2}']].set_index('Timestamp')
    frequency_history = df_filtered_history.resample('30T').size()
    
    # Plotting the frequency for each item in history
    plt.plot(frequency_history.index, frequency_history.values, marker='o', linestyle='-', label=f"{item_id} in History")

# Customize plot
plt.xlabel('Timestamp (30-minute intervals)')
plt.ylabel('Frequency')
plt.title('Frequency of Top 10 Impression-1 Items in History Over Time in Validation Impression')
plt.xticks(rotation=45)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05))
plt.tight_layout()

# Save the plot as a PNG file
top_100_items_plot_path = "top_10_Impress2History_frequency.png"
plt.savefig(top_100_items_plot_path)
plt.show()
