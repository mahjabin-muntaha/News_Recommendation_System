import tensorflow as tf
from tensorflow.keras.layers import Layer


class StackedAttentionLayer(Layer):
    def __init__(self, attention_size, iterations=3):
        super(StackedAttentionLayer, self).__init__()
        self.attention_size = attention_size
        self.iterations = iterations  # Apply attention multiple times

    def build(self, input_shape):
        self.W = self.add_weight(name="attention_weight",
                                 shape=(input_shape[-1], self.attention_size),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="attention_bias", shape=(self.attention_size,),
                                 initializer="zeros", trainable=True)
        self.u = self.add_weight(name="attention_context", shape=(self.attention_size, 1),
                                 initializer="glorot_uniform", trainable=True)

    def call(self, inputs):
        # Reshape input from (batch_size, embedding_dim) → (batch_size, embedding_dim, 1)
        expanded_inputs = tf.expand_dims(inputs, axis=1)  # Shape: (batch_size, 1, embedding_dim)

        attention_output = expanded_inputs  # Initialize input for iterations

        for _ in range(self.iterations):  # Apply attention multiple times
            score = tf.nn.tanh(tf.matmul(attention_output, self.W) + self.b)  # Compute attention scores
            attention_weights = tf.nn.softmax(tf.matmul(score, self.u), axis=1)  # Apply softmax
            attention_output = tf.reduce_sum(expanded_inputs * attention_weights, axis=1)  # Weighted sum

        return attention_output  # Shape: (batch_size, embedding_dim)
