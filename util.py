import pandas as pd
import os
import pickle


behaviors = pd.read_csv('behaviors_cleaned.csv')

behaviors['time'] = pd.to_datetime(behaviors['time'])
behaviors_sorted = behaviors.sort_values(by=['time'])

# Define train, validation, and test splits
total_records = len(behaviors_sorted)
train_end = int(0.8 * total_records)
val_end = int(0.9 * total_records)

train_set = behaviors_sorted.iloc[:train_end]
val_set = behaviors_sorted.iloc[train_end:val_end]
test_set = behaviors_sorted.iloc[val_end:]

# Ensure consistency: Check for unseen users and news articles in test/validation sets
train_users = set(train_set['user_id'])
train_news = set(
    news_id
    for history in train_set['history']
    for news_id in eval(history) if pd.notna(history)
)

# Filter validation and test sets for consistency
val_set = val_set[val_set['user_id'].isin(train_users)]
test_set = test_set[test_set['user_id'].isin(train_users)]

val_set = val_set[val_set['history'].apply(lambda x: all(news_id in train_news for news_id in eval(x)))]
test_set = test_set[test_set['history'].apply(lambda x: all(news_id in train_news for news_id in eval(x)))]

# Save the filtered data
train_file = 'datasets/train_set.csv'
val_file = 'datasets/val_set.csv'
test_file = 'datasets/test_set.csv'

train_set.to_csv(train_file, index=False)
val_set.to_csv(val_file, index=False)
test_set.to_csv(test_file, index=False)

def load_pickle(file_path):
    """ Utility function to load pickle files efficiently """
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None  # Return None if file doesn't exist

def save_pickle(obj, file_path):
    """ Utility function to save pickle files efficiently """
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

