import pandas as pd
from feature_engineering import FeatureEngineering
import os
import torch
from threadpoolctl import threadpool_limits

# Limit MKL threads to 1
with threadpool_limits(limits=1, user_api='blas'):
    # Check if the interaction matrix pickle file exists
    behaviors = pd.read_csv('behaviors_cleaned.csv')
    news = pd.read_csv('news_cleaned.csv')

    print(f'Behaviors: {behaviors.columns}')
    print(f'impressions: {behaviors["impressions"][0]}')

    feature_engineering = FeatureEngineering(behaviors, news)
    feature_engineering.run()
    pass



