import pandas as pd
import ast

# Step 1: Load data
results_df = pd.read_csv("results.csv")#, sep=",", header=None, names=["log_id", "clicked", "probs"])  # log_id, clicked, probs 파일
#print(results_df)
behaviors_df = pd.read_csv("behaviors.tsv", sep="\t", names=["log_id", "user_id", "timestamp", "history", "impression"])
#print(behaviors_df)
frequencies_df = pd.read_csv("Impression_frequencies2.csv")#, sep=",", names=["item_id", "Frequency"])  # Item_id, Frequency 파일
#print(frequencies_df)
# Step 2: Determine Popular Items (Top 50% Frequency)
frequency_threshold = frequencies_df["Frequency"].quantile(0.8) # frequencies_df["Frequency"].median()
popular_items = set(frequencies_df[frequencies_df["Frequency"] > frequency_threshold]["ItemID"])
unpopular_items = set(frequencies_df[frequencies_df["Frequency"] <= frequency_threshold]["ItemID"])

# Step 3: Map log_id to popular/unpopular
# behaviors_df에서 impression을 분리하고 Popular 여부 확인
def is_popular(log_impression):
    impression_items = log_impression.split()
    is_pop = any(item.split("-")[0] in popular_items for item in impression_items)
    return is_pop

behaviors_df["is_popular"] = behaviors_df["impression"].apply(is_popular)

# Step 4: Calculate correctness
def calculate_accuracy(row):
    clicked = ast.literal_eval(row["clicked"])  # Convert string to list
    probs = ast.literal_eval(row["probs"])      # Convert string to list

    # Get indices
    clicked_index = clicked.index(1) if 1 in clicked else -1
    max_prob_index = probs.index(max(probs))

    return clicked_index == max_prob_index

results_df["is_correct"] = results_df.apply(calculate_accuracy, axis=1)
print(results_df)
# Step 5: Merge results with behaviors
merged_df = pd.merge(results_df, behaviors_df, on="log_id")

# Step 6: Calculate accuracy for popular and unpopular items
popular_accuracy = merged_df[merged_df["is_popular"]]["is_correct"].mean()
unpopular_accuracy = merged_df[~merged_df["is_popular"]]["is_correct"].mean()
overall_accuracy = results_df["is_correct"].mean()
# Output the results
print(f"Popular Items Accuracy: {popular_accuracy:.2f}")
print(f"Unpopular Items Accuracy: {unpopular_accuracy:.2f}")
print(f"Overall Accuracy: {overall_accuracy:.2f}")