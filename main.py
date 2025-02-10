import pandas as pd
from feature_engineering import FeatureEngineering
from threadpoolctl import threadpool_limits
from model import NNModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import load_model
import tensorflow as tf
import global_variables
from sklearn.metrics import roc_auc_score
from sklearn.metrics import ndcg_score
import os
import pickle
import numpy as np

def mean_reciprocal_rank(y_true, y_pred):
    sorted_indices = np.argsort(-y_pred)  # Sort predictions in descending order
    ranks = np.where(y_true[sorted_indices] == 1)[0] + 1  # Get ranks of clicked articles
    return np.mean(1 / ranks) if len(ranks) > 0 else 0


# Limit MKL threads to 1
with threadpool_limits(limits=1, user_api='blas'):
    # # Check if the interaction matrix pickle file exists
    behaviors = pd.read_csv('./datasets/val_set.csv')
    news = pd.read_csv('news_cleaned.csv')
    file_path = global_variables.validation_data_path

    if os.path.exists(file_path + 'final_embeddings/user_embeddings.pkl'):
        with open(file_path + 'final_embeddings/user_embeddings.pkl', 'rb') as f:
            user_embeddings = pickle.load(f)
        with open(file_path + 'final_embeddings/news_embeddings.pkl', 'rb') as f:
            news_embeddings = pickle.load(f)
        with open(file_path + 'final_embeddings/labels.pkl', 'rb') as f:
            labels = pickle.load(f)
        with open(file_path + 'final_embeddings/user_embedding_dim.pkl', 'rb') as f:
            user_embedding_dim = pickle.load(f)
        with open(file_path + 'final_embeddings/news_embedding_dim.pkl', 'rb') as f:
            news_embedding_dim = pickle.load(f)
    else:
        feature_engineering = FeatureEngineering(behaviors, news, file_path)
        user_embedding_dim, news_embedding_dim, user_embeddings, news_embeddings, labels = feature_engineering.run()

    behaviors = pd.read_csv(file_path + 'behaviors_feature_engineered.csv')
    news = pd.read_csv(file_path + 'news_feature_engineered.csv')
    nnmodel = NNModel(user_embedding_dim, news_embedding_dim, user_embeddings, news_embeddings, labels)
    if os.path.exists('model_parameters/hybrid_model.weights.h5'):
        model = nnmodel.construct_model()
    else:
        model = nnmodel.fit(mode="save")

    # model = load_model("./model_parameters/hybrid_model.h5")
    user_embeddings = np.array(user_embeddings)
    news_embeddings = np.array(news_embeddings)
    labels = np.array(labels)


    predictions = model.predict([user_embeddings, news_embeddings], batch_size=256)
    print("Predictions min:", np.min(predictions))
    print("Predictions max:", np.max(predictions))
    print("Predictions mean:", np.mean(predictions))
    print("Labels unique values:", np.unique(labels, return_counts=True))
    auc_score = roc_auc_score(labels, predictions)
    print(f"AUC Score: {auc_score:.4f}")
    labels = labels.flatten()  # Convert from (4407,1) to (4407,)
    predictions = predictions.flatten()  # Convert from (4407,1) to (4407,)

    ndcg_value = ndcg_score([labels], [predictions])
    print(f"NDCG Score: {ndcg_value:.4f}")

    mrr_value = mean_reciprocal_rank(labels, predictions)
    print(f"MRR Score: {mrr_value:.4f}")

    pass



