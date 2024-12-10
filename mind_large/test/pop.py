import pandas as pd

# Load the Impression_frequencies2.csv to determine Popular and Unpopular items
frequencies_df = pd.read_csv("Impression_frequencies2.csv")

# Determine the threshold for Popular items (Top 50% Frequency)
frequency_threshold = frequencies_df["Frequency"].quantile(0.95) 
print("Threshold", frequency_threshold)
popular_items = set(frequencies_df[frequencies_df["Frequency"] > frequency_threshold]["ItemID"])
unpopular_items = set(frequencies_df[frequencies_df["Frequency"] <= frequency_threshold]["ItemID"])

# Load the behaviors.tsv file
behaviors_df = pd.read_csv("behaviors.tsv", sep="\t", names=["log_id", "user_id", "timestamp", "history", "impression"])

# Define a function to check if Impression contains Popular items
def check_popular(log_impression):
    impression_items = log_impression.split()
    # Check if any item ending with '-1' is in popular_items
    for item in impression_items:
        if item.endswith("-1") and item.split("-")[0] in popular_items:
            return 1  # Popular
    return 0  # Not Popular

# Apply the function to create a new column 'is_popular'
behaviors_df["is_popular"] = behaviors_df["impression"].apply(check_popular)

# Save the new DataFrame to a CSV file
behaviors_df[["log_id", "is_popular"]].to_csv("log_popularity_top_5.csv", index=False)

# Display the first few rows of the result for confirmation
behaviors_df[["log_id", "is_popular"]].head()
