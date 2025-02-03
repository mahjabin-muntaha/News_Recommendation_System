import os.path

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.models import load_model
import numpy as np

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
        combined = Concatenate()([user_input, news_input])
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

        model = Model(inputs=[user_input, news_input], outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        if os.path.exists("./model_parameters/hybrid_model.weights.h5"):
            model.load_weights("./model_parameters/hybrid_model.weights.h5")

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
            model.save("./model_parameters/hybrid_model.weights.h5")
        return model