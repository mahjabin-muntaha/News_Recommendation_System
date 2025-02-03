import pandas as pd
from feature_engineering import FeatureEngineering
from threadpoolctl import threadpool_limits
from model import NNModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import load_model
import global_variables
from sklearn.metrics import roc_auc_score
import os
import pickle
import numpy as np


# Limit MKL threads to 1
with threadpool_limits(limits=1, user_api='blas'):
    # # Check if the interaction matrix pickle file exists
    behaviors = pd.read_csv('./datasets/test_set.csv')
    news = pd.read_csv('news_cleaned.csv')
    file_path = global_variables.testing_data_path

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
    # if os.path.exists('model_parameters/hybrid_model.weights.h5'):
    #     model = nnmodel.construct_model()
    # else:
    #     model = nnmodel.fit(mode="save")
    model = load_model("./model_parameters/hybrid_model.h5")
    user_embeddings = np.array(user_embeddings)
    news_embeddings = np.array(news_embeddings)
    labels = np.array(labels)

    predictions = model.predict([user_embeddings, news_embeddings], batch_size=256)
    auc_score = roc_auc_score(labels, predictions)
    print(f"AUC Score: {auc_score:.4f}")

    pass



