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
from implicit.als import AlternatingLeastSquares

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate


class FeatureEngineering:
    def __init__(self, behaviors, news):
        self.subcategory_columns = None
        self.category_columns = None
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

    def combine_features(self, row):
        # Extract title and abstract embeddings
        title_emb = list(row['title_embedding'])  # Fixed-size (e.g., 768)
        abstract_emb = list(row['abstract_embedding'])  # Fixed-size (e.g., 768)

        # Ensure category and subcategory embeddings exist before accessing them
        category_emb = row[self.category_columns].values.tolist() if hasattr(self, 'category_columns') else [0] * len(
            self.category_columns)
        subcategory_emb = row[self.subcategory_columns].values.tolist() if hasattr(self, 'subcategory_columns') else [
                                                                                                                         0] * len(
            self.subcategory_columns)

        # Retrieve precomputed entity embeddings
        entity_emb = row['entity_embeddings']

        # Combine all features into a single vector
        combined = title_emb + abstract_emb + category_emb + subcategory_emb + entity_emb
        return combined

    # Compute and store entity embeddings in the DataFrame before apply()
    def compute_entity_embedding(self, entities, model):
        embeddings = [self.get_entity_embedding(e['Label'], model) for e in entities if
                      self.get_entity_embedding(e['Label'], model) is not None]
        return np.mean(embeddings, axis=0).tolist() if embeddings else [0] * 100  # Default zero vector

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
            self.news['title_embedding'] = embeddings['title_embedding']
            self.news['abstract_embedding'] = embeddings['abstract_embedding']
            self.news['combined_entities'] = self.news['combined_entities'].apply(literal_eval)
            all_entities = [[entity['Label'] for entity in entities] for entities in self.news['combined_entities']]
            model = Word2Vec(sentences=all_entities, vector_size=100, window=5, min_count=1, workers=4)
            self.news['entity_embeddings'] = self.news['combined_entities'].apply(
                lambda x: self.compute_entity_embedding(x, model))
            with open('news_features.pkl', 'wb') as f:
                pickle.dump(self.news, f)

            category_encoded = pd.get_dummies(self.news['category'], prefix='category')
            subcategory_encoded = pd.get_dummies(self.news['sub_category'], prefix='sub_category')
            self.news = pd.concat([self.news, category_encoded, subcategory_encoded], axis=1)
            self.category_columns = category_encoded.columns.tolist()
            self.subcategory_columns = subcategory_encoded.columns.tolist()

            self.news['combined_features'] = self.news.apply(self.combine_features, axis=1)
            with open('combined_features.pkl', 'wb') as f:
                pickle.dump(self.news[['news_id', 'combined_features']], f)

        if os.path.exists('user_features.pkl'):
            with open('user_features.pkl', 'rb') as f:
                user_profiles = pickle.load(f)
        else:
            self.behaviors['history'] = self.behaviors['history'].apply(eval)
            news_embeddings_map = {
                row['news_id']: np.concatenate([row['title_embedding'], row['abstract_embedding']])
                for _, row in embeddings.iterrows()
            }
            user_profiles = {}
            for _, row in self.behaviors.iterrows():
                user_id = row['user_id']
                clicked_articles = row['history']
                valid_embeddings = [news_embeddings_map[article_id] for article_id in clicked_articles if
                                    article_id in news_embeddings_map]
                if valid_embeddings:
                    user_profiles[user_id] = np.mean(valid_embeddings, axis=0)
            with open('user_features.pkl', 'wb') as f:
                pickle.dump(user_profiles, f)

        if os.path.exists('user_cf_embeddings.pkl') and os.path.exists('news_cf_embeddings.pkl'):
            with open('user_cf_embeddings.pkl', 'rb') as f:
                user_cf_embeddings = pickle.load(f)
            with open('news_cf_embeddings.pkl', 'rb') as f:
                news_cf_embeddings = pickle.load(f)
        else:
            interaction_matrix = int_matrix.T
            als = AlternatingLeastSquares(factors=128, regularization=0.1, iterations=20)
            als.fit(interaction_matrix.T)
            user_cf_embeddings = als.user_factors
            news_cf_embeddings = als.item_factors
            with open('user_cf_embeddings.pkl', 'wb') as f:
                pickle.dump(user_cf_embeddings, f)
            with open('news_cf_embeddings.pkl', 'wb') as f:
                pickle.dump(news_cf_embeddings, f)

        user_ids = set(self.behaviors['user_id'])
        news_ids = set()
        for history in self.behaviors['history']:
            news_ids.update(ast.literal_eval(history))

        unique_users = sorted(set(user_ids))
        unique_news = sorted(set(news_ids))
        user_id_to_index = {user_id: idx for idx, user_id in enumerate(unique_users)}
        news_id_to_index = {news_id: idx for idx, news_id in enumerate(unique_news)}
        filtered_news_id_to_index = {news_id: idx for news_id, idx in news_id_to_index.items() if
                                     idx < len(news_cf_embeddings)}

        news_cf_map = {news_id: news_cf_embeddings[idx] for news_id, idx in filtered_news_id_to_index.items()}
        user_cf_map = {user_id: user_cf_embeddings[idx] for user_id, idx in user_id_to_index.items()}

        with open('combined_features.pkl', 'rb') as f:
            content_embeddings = pickle.load(f)

        content_embeddings_dict = content_embeddings.set_index('news_id')['combined_features'].to_dict()
        hybrid_user_embeddings = {user_id: np.concatenate([user_cf_map[user_id], user_profiles[user_id]]) for user_id in
                                  user_cf_map if user_id in user_profiles}
        hybrid_news_embeddings = {news_id: np.concatenate([news_cf_map[news_id], content_embeddings_dict[news_id]]) for
                                  news_id in news_cf_map if news_id in content_embeddings_dict}

        user_embeddings = []
        news_embeddings = []
        labels = []

        self.behaviors['impressions'] = self.behaviors['impressions'].apply(literal_eval)
        for _, row in self.behaviors.iterrows():
            user_id = row['user_id']
            impressions = row['impressions']
            for news_id, label in impressions:
                if user_id in hybrid_user_embeddings and news_id in hybrid_news_embeddings:
                    user_embeddings.append(hybrid_user_embeddings[user_id])
                    news_embeddings.append(hybrid_news_embeddings[news_id])
                    labels.append(label)

        for idx, emb in enumerate(news_embeddings):
            if len(emb) != len(news_embeddings[0]):
                print(f"Inconsistent shape at index {idx}: {len(emb)}")

        user_embedding_dim = len(next(iter(hybrid_user_embeddings.values())))
        news_embedding_dim = len(next(iter(hybrid_news_embeddings.values())))

        user_input = Input(shape=(user_embedding_dim,), name="user_input")
        news_input = Input(shape=(news_embedding_dim,), name="news_input")
        combined = Concatenate()([user_input, news_input])
        x = Dense(128, activation='relu')(combined)
        x = Dense(64, activation='relu')(x)
        output = Dense(1, activation='sigmoid', name="output")(x)

        model = Model(inputs=[user_input, news_input], outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        user_embeddings = np.array(user_embeddings)
        news_embeddings = np.array(news_embeddings)
        labels = np.array(labels)

        model.fit([user_embeddings, news_embeddings], labels, batch_size=256, epochs=10, validation_split=0.1)