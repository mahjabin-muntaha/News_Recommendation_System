import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import nltk
import json

nltk.download('stopwords')
nltk.download('punkt')
nltk.download("punkt_tab")

def parse_impressions(impressions_str):
    if isinstance(impressions_str, str) and impressions_str.strip():
        return [tuple(item.split('-')) for item in impressions_str.strip().split(' ')]
    return []

def clean_text(text):
    # remove html tags
    text = re.sub(r'<.*?>', '', text)
    # remove only special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # normalize to lowercase
    text = text.lower()
    # remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    text = ' '.join([word for word in tokens if word not in stop_words])
    return text

def extract_entities(entity_field):
    if pd.isnull(entity_field) or not entity_field.strip():
        return []
    try:
        entities = json.loads(entity_field)
        # return [entity['Label'] for entity in entities if 'Label' in entity]
        extracted = [{'Label': entity['Label'], 'Type': entity['Type'], 'Confidence': entity['Confidence']} for entity
                     in entities]
        return extracted
    except json.JSONDecodeError:
        return []

def combine_entities(title_entities, abstract_entities):
    combined = title_entities + abstract_entities  # Combine the two lists
    # Optional: Deduplicate based on 'Label' and 'Type'
    unique_combined = {f"{entity['Label']}_{entity['Type']}": entity for entity in combined}.values()
    return list(unique_combined)

behaviors = pd.read_csv('./MIND_dataset/MINDsmall_train/behaviors.tsv', sep='\t', names=['impression_id', 'user_id', 'time', 'history', 'impressions'])
news = pd.read_csv('./MIND_dataset/MINDsmall_train/news.tsv', sep='\t', names=['news_id', 'category', 'sub_category', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])

print(f'Behaviors: {behaviors.shape}')
print(f'news: {news.shape}')

# Drop rows with missing values
behaviors_cleaned = behaviors.dropna(subset=['user_id', 'history', 'impressions'])

# Split the history column into a list of news IDs
behaviors_cleaned.loc[:, 'history'] = behaviors_cleaned['history'].fillna('').apply(lambda x: x.split(' '))

behaviors_cleaned.loc[:, 'impressions'] = behaviors_cleaned['impressions'].apply(parse_impressions)

news_cleaned = news.dropna(subset=['title', 'category', 'sub_category', 'abstract'])

print("before cleaning")
print(news_cleaned[['title']].head())

news_cleaned.loc[:, 'title'] = news_cleaned['title'].apply(clean_text)
news_cleaned.loc[:, 'abstract'] = news_cleaned['abstract'].apply(clean_text)

# Test the extract_entities function on a single example
news_cleaned.loc[:, 'title_entities_extracted'] = news_cleaned['title_entities'].apply(extract_entities)
news_cleaned.loc[:, 'abstract_entities_extracted'] = news_cleaned['abstract_entities'].apply(extract_entities)
news_cleaned.loc[:, 'combined_entities'] = news_cleaned.apply(
    lambda row: combine_entities(row['title_entities_extracted'], row['abstract_entities_extracted']),
    axis=1
)

behaviors_cleaned.to_csv('behaviors_cleaned.csv', index=False)
news_cleaned.to_csv('news_cleaned.csv', index=False)