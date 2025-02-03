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
from util import load_pickle, save_pickle

class FeatureEngineering:
    def __init__(self, behaviors, news, location):
        self.subcategory_columns = None
        self.category_columns = None
        self.behaviors = behaviors
        self.news = news
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = AutoModel.from_pretrained('distilbert-base-uncased')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.file_path = location

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

    def encode_texts_with_bert(self, texts, batch_size=16):
        """ Efficiently encode texts using DistilBERT in smaller batches """
        self.model.eval()  # Set model to evaluation mode
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]  # Create batch
            inputs = self.tokenizer(batch, return_tensors='pt', truncation=True, padding=True, max_length=64)
            inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}  # Move to GPU if available

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(
                    dim=1).cpu().numpy()  # Move to CPU before NumPy conversion

            embeddings.extend(batch_embeddings)

        return np.array(embeddings)  # Convert list to NumPy array

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

    def compute_entity_embedding(self, entities, model):
        valid_embeddings = np.array([
            model.wv[e['Label']] for e in entities if e['Label'] in model.wv
        ])
        return valid_embeddings.mean(axis=0).tolist() if len(valid_embeddings) > 0 else [0] * 100  # Zero vector fallback

    def get_hybrid_embeddings(self, user_ids, news_ids, user_cf_embeddings, news_cf_embeddings, user_profiles):
        unique_users = sorted(set(user_ids))
        unique_news = sorted(set(news_ids))
        user_id_to_index = {user_id: idx for idx, user_id in enumerate(unique_users)}
        news_id_to_index = {news_id: idx for idx, news_id in enumerate(unique_news)}
        filtered_news_id_to_index = {news_id: idx for news_id, idx in news_id_to_index.items() if
                                     idx < len(news_cf_embeddings)}

        news_cf_map = {news_id: news_cf_embeddings[idx] for news_id, idx in filtered_news_id_to_index.items()}
        user_cf_map = {user_id: user_cf_embeddings[idx] for user_id, idx in user_id_to_index.items()}
        content_embeddings = load_pickle(self.file_path + 'combined_features.pkl')

        content_embeddings_dict = content_embeddings.set_index('news_id')['combined_features'].to_dict()
        hybrid_user_embeddings = {user_id: np.concatenate([user_cf_map[user_id], user_profiles[user_id]]) for user_id in
                                  user_cf_map if user_id in user_profiles}
        alpha = 0.7

        hybrid_news_embeddings = {news_id: np.concatenate([news_cf_map[news_id], content_embeddings_dict[news_id]])
                                  for news_id in news_cf_map if news_id in content_embeddings_dict}

        return hybrid_user_embeddings, hybrid_news_embeddings

    def load_data(self, hybrid_user_embeddings, hybrid_news_embeddings):
        if os.path.exists(self.file_path + 'final_embeddings/user_embeddings.pkl') and os.path.exists(
                self.file_path + 'final_embeddings/news_embeddings.pkl'):
            user_embeddings = load_pickle(self.file_path + 'final_embeddings/user_embeddings.pkl')
            news_embeddings = load_pickle(self.file_path + 'final_embeddings/news_embeddings.pkl')
            labels = load_pickle(self.file_path + 'final_embeddings/labels.pkl')
        else:
            user_embeddings = []
            news_embeddings = []
            labels = []
            for _, row in self.behaviors.iterrows():
                user_id = row['user_id']
                impressions = row['impressions']
                for news_id, label in impressions:
                    if user_id in hybrid_user_embeddings and news_id in hybrid_news_embeddings:
                        user_embeddings.append(hybrid_user_embeddings[user_id])
                        news_embeddings.append(hybrid_news_embeddings[news_id])
                        labels.append(label)
            labels = list(map(int, labels))
            save_pickle(user_embeddings, self.file_path + 'final_embeddings/user_embeddings.pkl')
            save_pickle(news_embeddings, self.file_path + 'final_embeddings/news_embeddings.pkl')
            save_pickle(labels, self.file_path + 'final_embeddings/labels.pkl')

        return user_embeddings, news_embeddings, labels

    def run(self):
        if os.path.exists(self.file_path + 'interaction_matrix.pkl'):
            int_matrix = load_pickle(self.file_path + 'interaction_matrix.pkl')
        else:
            int_matrix = self.create_interaction_matrix()
            save_pickle(int_matrix, self.file_path + 'interaction_matrix.pkl')

        if os.path.exists(self.file_path + 'embeddings.pkl'):
            embeddings = load_pickle(self.file_path + 'embeddings.pkl')
        else:
            self.news['abstract'] = self.news['abstract'].apply(self.preprocess_text)
            title_embeddings = self.encode_texts_with_bert(self.news['title'].tolist())
            self.news['title_embedding'] = list(title_embeddings)  # Convert to list

            abstract_embeddings = self.encode_texts_with_bert(self.news['abstract'].tolist())
            self.news['abstract_embedding'] = list(abstract_embeddings)  # Convert to list
            embeddings = self.news[['news_id', 'title_embedding', 'abstract_embedding']]
            save_pickle(embeddings, self.file_path + 'embeddings.pkl')

        if os.path.exists(self.file_path + 'news_features.pkl'):
            news_features = load_pickle(self.file_path + 'news_features.pkl')
        else:
            self.news['title_embedding'] = embeddings['title_embedding']
            self.news['abstract_embedding'] = embeddings['abstract_embedding']
            self.news['combined_entities'] = self.news['combined_entities'].apply(literal_eval)
            all_entities = [[entity['Label'] for entity in entities] for entities in self.news['combined_entities']]
            model = Word2Vec(sentences=all_entities, vector_size=100, window=5, min_count=1, workers=4)
            self.news['entity_embeddings'] = self.news['combined_entities'].apply(
                lambda x: self.compute_entity_embedding(x, model))
            save_pickle(self.news[['news_id', 'entity_embeddings']], self.file_path + 'news_features.pkl')

            category_encoded = pd.get_dummies(self.news['category'], prefix='category')
            subcategory_encoded = pd.get_dummies(self.news['sub_category'], prefix='sub_category')
            self.news = pd.concat([self.news, category_encoded, subcategory_encoded], axis=1)
            self.category_columns = category_encoded.columns.tolist()
            self.subcategory_columns = subcategory_encoded.columns.tolist()

            self.news['combined_features'] = self.news.apply(self.combine_features, axis=1)
            save_pickle(self.news[['news_id', 'combined_features']], self.file_path + 'combined_features.pkl')

        if os.path.exists(self.file_path + 'user_features.pkl'):
            user_profiles = load_pickle(self.file_path + 'user_features.pkl')
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
            save_pickle(user_profiles, self.file_path + 'user_features.pkl')

        if os.path.exists(self.file_path + 'user_cf_embeddings.pkl') and os.path.exists(
                self.file_path + 'news_cf_embeddings.pkl'):
            user_cf_embeddings = load_pickle(self.file_path + 'user_cf_embeddings.pkl')
            news_cf_embeddings = load_pickle(self.file_path + 'news_cf_embeddings.pkl')
        else:
            interaction_matrix = int_matrix
            als = AlternatingLeastSquares(
                factors=128,
                regularization=0.05,
                iterations=15,
                use_gpu=False
            )

            als.fit(interaction_matrix.astype(np.float32))
            user_cf_embeddings = als.user_factors
            news_cf_embeddings = als.item_factors
            save_pickle(user_cf_embeddings, self.file_path + 'user_cf_embeddings.pkl')
            save_pickle(news_cf_embeddings, self.file_path + 'news_cf_embeddings.pkl')

        self.behaviors['impressions'] = self.behaviors['impressions'].apply(literal_eval)

        self.behaviors.to_csv(self.file_path + 'behaviors_feature_engineered.csv', index=False)
        self.news.to_csv(self.file_path + 'news_feature_engineered.csv', index=False)

        user_ids = set(self.behaviors['user_id'])
        news_ids = set()
        for history in self.behaviors['history']:
            news_ids.update(ast.literal_eval(history))

        hybrid_user_embeddings, hybrid_news_embeddings = self.get_hybrid_embeddings(user_ids, news_ids,
                                                                                    user_cf_embeddings,
                                                                                    news_cf_embeddings, user_profiles)

        user_embedding_dim = len(next(iter(hybrid_user_embeddings.values())))
        news_embedding_dim = len(next(iter(hybrid_news_embeddings.values())))
        save_pickle(news_embedding_dim, self.file_path + 'final_embeddings/news_embedding_dim.pkl')
        save_pickle(user_embedding_dim, self.file_path + 'final_embeddings/user_embedding_dim.pkl')
        user_embeddings, news_embeddings, labels = self.load_data(hybrid_user_embeddings, hybrid_news_embeddings)
        return user_embedding_dim, news_embedding_dim, user_embeddings, news_embeddings, labels

