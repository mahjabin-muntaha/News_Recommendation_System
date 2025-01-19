from ast import literal_eval

import pandas as pd
import ast
from scipy.sparse import csr_matrix
import pickle
from transformers import AutoTokenizer, AutoModel
import torch
from gensim.models import Word2Vec
import os
import numpy as np
import implicit


class FeatureEngineering:
    def __init__(self, behaviors, news):
        self.behaviors = behaviors
        self.news = news
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = AutoModel.from_pretrained('distilbert-base-uncased')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def create_interaction_matrix(self):
        user_ids = []
        news_ids = []
        labels = []
        for _, row in self.behaviors.iterrows():
            user = row['user_id']
            if isinstance(row['impressions'], str):
                # Safely evaluate the string to convert it into a Python list
                impressions = ast.literal_eval(row['impressions'])
            else:
                impressions = row['impressions']
            for news_id, label in impressions:
                news_ids.append(news_id)
                labels.append(int(label))
                user_ids.append(user)
        unique_users = {user: idx for idx, user in enumerate(sorted(set(user_ids)))}
        unique_news = {news: idx for idx, news in enumerate(sorted(set(news_ids)))}

        user_indices = [unique_users[user] for user in user_ids]
        news_indices = [unique_news[news] for news in news_ids]

        num_users = len(unique_users)
        num_news = len(unique_news)

        interaction_matrix = csr_matrix(
            (labels, (user_indices, news_indices)),
            shape=(num_users, num_news)
        )
        return interaction_matrix

    def encode_with_bert(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=64)
        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}  # Move all tensors to GPU
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Compute the mean embedding and move it to CPU before converting to NumPy
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding

    def get_entity_embedding(self, entity, model):
        return model.wv[entity] if entity in model.wv else None

    def preprocess_text(self, text):
        if pd.isnull(text):  # Check for NaN or None
            return ""
        return str(text)  # Ensure the text is a string

    def run(self):
        if os.path.exists('interaction_matrix.pkl'):
            with open('interaction_matrix.pkl', 'rb') as f:
                int_matrix = pickle.load(f)
        else:
            int_matrix = self.create_interaction_matrix()
            with open('interaction_matrix.pkl', 'wb') as f:
                pickle.dump(int_matrix, f)

        if os.path.exists('embeddings.pkl'):
            with open('embeddings.pkl', 'rb') as f:
                embeddings = pickle.load(f)
        else:
            # Apply preprocessing to the abstract column before tokenization
            self.news['abstract'] = self.news['abstract'].apply(self.preprocess_text)
            self.news['title_embedding'] = self.news['title'].apply(self.encode_with_bert)
            self.news['abstract_embedding'] = self.news['abstract'].apply(self.encode_with_bert)
            embeddings = self.news[['title_embedding', 'abstract_embedding']]
            with open('embeddings.pkl', 'wb') as f:
                pickle.dump(embeddings, f)
        if os.path.exists('news_features.pkl'):
            with open('news_features.pkl', 'rb') as f:
                news_features = pickle.load(f)
        else:
            self.news['combined_entities'] = self.news['combined_entities'].apply(literal_eval)
            all_entities = [[entity['Label'] for entity in entities] for entities in self.news['combined_entities']]

            # Train a Word2Vec model with adjusted parameters
            model = Word2Vec(sentences=all_entities, vector_size=100, window=5, min_count=1, workers=4)

            self.news['entity_embeddings'] = self.news['combined_entities'].apply(
                lambda entities: [self.get_entity_embedding(e['Label'], model) for e in entities if
                                  self.get_entity_embedding(e['Label'], model) is not None]
            )

            category_encoded = pd.get_dummies(self.news['category'], prefix='category')
            subcategory_encoded = pd.get_dummies(self.news['sub_category'], prefix='sub_category')

            # Concatenate with the original DataFrame
            self.news = pd.concat([self.news, category_encoded, subcategory_encoded], axis=1)
            self.news['title_embedding'] = embeddings['title_embedding']
            self.news['abstract_embedding'] = embeddings['abstract_embedding']
            with open('news_features.pkl', 'wb') as f:
                pickle.dump(self.news, f)
            self.news['combined_features'] = self.news.apply(
                lambda row: list(row['title_embedding']) +
                            list(row['abstract_embedding']) +
                            list(row[category_encoded.columns]) +
                            list(row[subcategory_encoded.columns]) +
                            [item for sublist in row['entity_embeddings'] for item in sublist],
                axis=1
            )
            with open('combined_features.pkl', 'wb') as f:
                pickle.dump(self.news['combined_features'], f)

        if os.path.exists('user_features.pkl'):
            with open('user_features.pkl', 'rb') as f:
                user_features = pickle.load(f)
        else:
            self.behaviors['history'] = self.behaviors['history'].apply(eval)
            news_embeddings_map = {
                row['news_id']: np.concatenate([row['title_embedding'], row['abstract_embedding']])
                for _, row in embeddings.iterrows()
            }

            # Compute user embeddings
            user_profiles = {}
            for _, row in self.behaviors.iterrows():
                user_id = row['user_id']
                clicked_articles = row['history']

                # Retrieve valid embeddings
                valid_embeddings = [
                    news_embeddings_map[article_id]
                    for article_id in clicked_articles if article_id in news_embeddings_map
                ]

                if valid_embeddings:
                    user_profiles[user_id] = np.mean(valid_embeddings, axis=0)

            with open('user_features.pkl', 'wb') as f:
                pickle.dump(user_profiles, f)

        interaction_matrix = int_matrix.T
        # Train ALS model
        model = implicit.als.AlternatingLeastSquares(
            factors=128,  # Number of latent factors
            regularization=0.1,
            iterations=20
        )
        model.fit(interaction_matrix)

        # User and item embeddings
        user_factors = model.user_factors  # Shape: (num_users, factors)
        item_factors = model.item_factors  # Shape: (num_items, factors)