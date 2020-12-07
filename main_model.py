# inbuilt lib imports:
from typing import List, Dict, Tuple
import os

# external libs
import tensorflow as tf
from tensorflow.keras import layers, models


class MainClassifier(models.Model):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 filters: List[int],
                 out_channels: int,
                 drop_prob: float,
                 hidden_units,
                 num_classes: int = 2
                 ) -> 'MainClassifier':

        super(MainClassifier, self).__init__()

        self.embedding_dim = embedding_dim
        self._embeddings = tf.Variable(tf.random.normal((vocab_size, embedding_dim)), trainable=True)

        self.filters = filters
        self.out_channels = out_channels
        self.conv1 = layers.Convolution2D(filters=out_channels, kernel_size=[filters[0], embedding_dim], activation='relu')
        self.conv2 = layers.Convolution2D(filters=out_channels, kernel_size=[filters[1], embedding_dim], activation='relu')
        self.conv3 = layers.Convolution2D(filters=out_channels, kernel_size=[filters[2], embedding_dim], activation='relu')

        self.mlp1_layer = layers.Dense(units=hidden_units, activation='tanh')
        self.mlp2_layer = layers.Dense(units=50, activation='tanh')
        self._classification_layer = layers.Dense(units=num_classes)

        self.dropout = layers.Dropout(drop_prob)


    def call(self,
             inputs: tf.Tensor,
             training=False):

        batch_size, max_tokens = inputs.shape
        word_embed = tf.nn.embedding_lookup(self._embeddings, inputs)

        # num_slides = max_tokens - (self.filters[0] - 1)
        conv_input = tf.reshape(word_embed, [batch_size, max_tokens, self.embedding_dim, 1])
        conv1_output = self.conv1(conv_input)
        max_pooled_output_1 = tf.nn.max_pool(conv1_output, ksize=[1, max_tokens - (self.filters[0] - 1), 1, 1],
                                             strides=[1, 1, 1, 1], padding='VALID')

        conv2_output = self.conv2(conv_input)
        max_pooled_output_2 = tf.nn.max_pool(conv2_output, ksize=[1, max_tokens - (self.filters[1] - 1), 1, 1],
                                             strides=[1, 1, 1, 1], padding='VALID')

        conv3_output = self.conv3(conv_input)
        max_pooled_output_3 = tf.nn.max_pool(conv3_output, ksize=[1, max_tokens - (self.filters[2] - 1), 1, 1],
                                             strides=[1, 1, 1, 1], padding='VALID')

        cnn_output = tf.concat([max_pooled_output_1, max_pooled_output_2, max_pooled_output_3], -1)
        cnn_output_dim = len(self.filters) * self.out_channels
        cnns_concat_output = tf.reshape(cnn_output, [-1, cnn_output_dim])

        mlp1_output = self.mlp1_layer(cnns_concat_output)
        mlp1_output = self.dropout(mlp1_output, training=training)

        mlp2_output = self.mlp2_layer(mlp1_output)
        mlp2_output = self.dropout(mlp2_output, training=training)

        logits = self._classification_layer(mlp2_output)
        return {"logits": logits}