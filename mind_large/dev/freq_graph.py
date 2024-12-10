import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


df = pd.read_csv('behaviors.tsv', sep="\t", names=["Index", "UserID", "Timestamp", "History", "Impression"])

# Converting the 'Timestamp' column to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
item_id = 'N47061'
print(item_id)
# print(df['History'])
# Filtering impressions to check for occurrences of 'N55689-1'
df[f'Contains_{item_id}-1'] = df['Impression'].apply(lambda x: f'{item_id}-1' in x)
df[f'Contains_{item_id}-0'] = df['Impression'].apply(lambda x: f'{item_id}-0' in x)
#print(df['History'])

df[f'History_Contains_{item_id}'] = df['History'].apply(lambda x: item_id in x if isinstance(x, str) else False)

# Filtering rows where 'N55689-1' is in the impression
filtered_df = df[df[f'Contains_{item_id}-1']]

# Resampling the frequency based on 30-minute intervals
filtered_df.set_index('Timestamp', inplace=True)
frequency_series = filtered_df.resample('30T').size()

# Plotting the frequency with different colors for each condition
plt.figure(figsize=(10, 6))

# Original N55689-1 impression plot (blue)
plt.plot(frequency_series.index, frequency_series.values, marker='o', linestyle='-', color='blue', label=f"{item_id}-1 in Impression")
filtered_df_N55689_0 = df[df[f'Contains_{item_id}-0']].set_index('Timestamp')
frequency_N55689_0 = filtered_df_N55689_0.resample('30T').size()

filtered_df_history = df[df[f'History_Contains_{item_id}']].set_index('Timestamp')
frequency_history_N55689 = filtered_df_history.resample('30T').size()
# N55689-0 impression plot (red)
plt.plot(frequency_N55689_0.index, frequency_N55689_0.values, marker='o', linestyle='-', color='red', label=f"{item_id}-0 in Impression")

# N55689 in history plot (yellow)
plt.plot(frequency_history_N55689.index, frequency_history_N55689.values, marker='o', linestyle='-', color='yellow', label=f"{item_id} in History")

# Adding labels, title, and legend
plt.xlabel('Timestamp (30-minute intervals)')
plt.ylabel('Frequency')
plt.title(f'Frequency of {item_id} Variants in Impressions and History Over Time in Validation Sets')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

# Save the overlaid plot as a PNG file
overlay_plot_path = f"overlay_frequency_{item_id}_impression_history.png"
plt.savefig(overlay_plot_path)