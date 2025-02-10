import os.path

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization
import numpy as np
from attention_layer import StackedAttentionLayer

def bpr_loss(y_true, y_pred):
    """ Bayesian Personalized Ranking (BPR) Loss """
    y_pred = tf.squeeze(y_pred)  # Ensure predictions are 1D
    y_true = tf.cast(y_true, tf.float32)  # Ensure labels are float

    positive_scores = y_pred * y_true  # Scores for clicked items
    negative_scores = y_pred * (1 - y_true)  # Scores for non-clicked items

    # Compute BPR loss
    loss = -tf.reduce_mean(tf.math.log_sigmoid(positive_scores - negative_scores + 1e-10))  # Stability with epsilon
    return loss

class NNModel:
    def __init__(self, user_embedding_dim, news_embedding_dim, user_embeddings, news_embeddings, labels):
        self.user_embedding_dim = user_embedding_dim
        self.news_embedding_dim = news_embedding_dim
        self.user_embeddings = user_embeddings
        self.news_embeddings = news_embeddings
        self.labels = labels

    def construct_model(self):
        user_input = Input(shape=(self.user_embedding_dim,), name="user_input")
        news_input = Input(shape=(self.news_embedding_dim,), name="news_input")

        # Apply Attention Layer (Now working with 2D input)
        user_att = StackedAttentionLayer(attention_size=128, iterations=1)(user_input)
        news_att = StackedAttentionLayer(attention_size=128, iterations=1)(news_input)

        # Concatenate attended embeddings
        combined = Concatenate()([user_att, news_att])

        # Fully connected layers
        x = Dense(256, activation='relu')(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        output = Dense(1, activation='sigmoid', name="output")(x)

        model = tf.keras.models.Model(inputs=[user_input, news_input], outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def fit(self, mode):
        model = self.construct_model()
        user_embeddings = np.array(self.user_embeddings)
        news_embeddings = np.array(self.news_embeddings)
        labels = np.array(self.labels)
        train_dataset = tf.data.Dataset.from_tensor_slices(
            ({"user_input": user_embeddings, "news_input": news_embeddings}, labels)
        ).batch(256).prefetch(tf.data.AUTOTUNE)
        model.fit(train_dataset, epochs=10)
        if mode == "save":
            model.save("./model_parameters/hybrid_model.h5")
            model.save_weights("./model_parameters/hybrid_model.weights.h5")
        return model