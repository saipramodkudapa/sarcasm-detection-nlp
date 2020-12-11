# inbuilt lib imports:
from typing import List, Dict, Tuple
import os

# external libs
import tensorflow as tf
from tensorflow.keras import layers, models
from transformers import TFBertModel
from transformers import logging
logging.set_verbosity_error()


# base CNN model used by both only_CNN and CNN_GRU models
class baseCNNmodel(models.Model):
    def __init__(self,
                 embedding_dim: int,
                 filters: List[int],
                 out_channels: int
                 ):
        super(baseCNNmodel, self).__init__()

        # define the three convolutional layers with respective filter sizes
        self.filters = filters
        self.out_channels = out_channels
        self.conv1 = layers.Convolution2D(filters=out_channels, kernel_size=[filters[0], embedding_dim], activation='relu')
        self.conv2 = layers.Convolution2D(filters=out_channels, kernel_size=[filters[1], embedding_dim], activation='relu')
        self.conv3 = layers.Convolution2D(filters=out_channels, kernel_size=[filters[2], embedding_dim], activation='relu')

    def call(self,
             conv_input: tf.Tensor):

        _, max_tokens, _, _ = conv_input.shape
        conv1_output = self.conv1(conv_input)

        # number of window slides = max_tokens - (self.filters[0] - 1)
        # Pool on the second dimension having
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

        return cnns_concat_output


class onlyCNNmodel(models.Model):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 filters: List[int],
                 out_channels: int,
                 drop_prob: float,
                 nn_hidden_dim: int,
                 num_classes: int = 2
                 ):
        super(onlyCNNmodel, self).__init__()

        self._baseCNN = baseCNNmodel(embedding_dim, filters, out_channels)

        self.embedding_dim = embedding_dim
        self._embeddings = tf.Variable(tf.random.normal((vocab_size, embedding_dim)), trainable=True)

        self.mlp1_layer = layers.Dense(units=nn_hidden_dim, activation='tanh')
        self.mlp2_layer = layers.Dense(units=50, activation='tanh')
        
        self._classification_layer = layers.Dense(units=num_classes)

        self.dropout = layers.Dropout(drop_prob)

    def call(self,
             inputs: tf.Tensor,
             training=False):

        batch_size, max_tokens = inputs.shape
        word_embed = tf.nn.embedding_lookup(self._embeddings, inputs)

        conv_input = tf.reshape(word_embed, [batch_size, max_tokens, self.embedding_dim, 1])

        cnns_concat_output =  self._baseCNN(conv_input)

        mlp1_output = self.mlp1_layer(cnns_concat_output)
        mlp1_output = self.dropout(mlp1_output, training=training)

        mlp2_output = self.mlp2_layer(mlp1_output)
        mlp2_output = self.dropout(mlp2_output, training=training)

        logits = self._classification_layer(mlp2_output)
        return {"logits": logits}        


class CNNandAttentiveBiGRUmodel(models.Model):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 filters: List[int],
                 out_channels: int,
                 drop_prob: float,
                 nn_hidden_dim: int,
                 gru_hidden_dim: 128,
                 num_classes: int = 2
                 ):
        super(CNNandAttentiveBiGRUmodel, self).__init__()

        self._baseCNN = baseCNNmodel(embedding_dim, filters, out_channels)
      
        self.embedding_dim = embedding_dim
        self._embeddings = tf.Variable(tf.random.normal((vocab_size, embedding_dim)), trainable=True)

        self.omegas = tf.Variable(tf.random.normal((gru_hidden_dim * 2, 1)))
        gru = layers.GRU(units=gru_hidden_dim, return_sequences=True)
        self.bi_gru_layer = layers.Bidirectional(layer=gru, merge_mode='concat')

        self.mlp1_layer = layers.Dense(units=nn_hidden_dim, activation='tanh')
        self.mlp2_layer = layers.Dense(units=50, activation='tanh')
        
        self._classification_layer = layers.Dense(units=num_classes)

        self.dropout = layers.Dropout(drop_prob)

    def attn(self, H):
        M = tf.nn.tanh(H)
        dot_product = tf.tensordot(M, self.omegas, axes=[-1, 0])
        alpha = tf.nn.softmax(dot_product, axis=1)
        r = tf.reduce_sum(tf.multiply(H, alpha), axis=1)
        output = tf.nn.tanh(r)
        return output

    def call(self,
             inputs: tf.Tensor,
             training=False):

        batch_size, max_tokens = inputs.shape
        word_embed = tf.nn.embedding_lookup(self._embeddings, inputs)
        sequence_mask = tf.cast(inputs != 0, tf.float32)
        gru_output = self.bi_gru_layer(word_embed, mask=sequence_mask)
        attn_output = self.attn(gru_output)

        # num_slides = max_tokens - (self.filters[0] - 1)
        conv_input = tf.reshape(word_embed, [batch_size, max_tokens, self.embedding_dim, 1])
        
        cnns_concat_output = self._baseCNN(conv_input)

        cnn_gru_concat = tf.concat([attn_output, cnns_concat_output], -1)

        mlp1_output = self.mlp1_layer(cnn_gru_concat)
        mlp1_output = self.dropout(mlp1_output, training=training)

        mlp2_output = self.mlp2_layer(mlp1_output)
        mlp2_output = self.dropout(mlp2_output, training=training)

        logits = self._classification_layer(mlp2_output)
        return {"logits": logits}


def create_bert_cnn_model(num_tokens: int, num_filters: int, filter_size: int, embedding_dim: int, nn_hidden_dim: int, dropout_prob: float):
    # define the encoder for bert model
    bert_encoder = TFBertModel.from_pretrained('bert-base-uncased')

    input_word_ids = tf.keras.Input(shape=(num_tokens,), dtype=tf.int32, name="input_word_ids")
    bert_embedding = bert_encoder([input_word_ids])
    cnn_input = tf.expand_dims(bert_embedding[0], -1)
    cnn_output = layers.Convolution2D(filters=num_filters, kernel_size=[filter_size, embedding_dim], activation='relu')(cnn_input)
    max_pooled_output = tf.nn.max_pool(cnn_output, ksize=[1, 13, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
    max_pooled_output = tf.reshape(max_pooled_output, [-1, 200])
    hidden_output = layers.Dense(nn_hidden_dim, activation='relu')(max_pooled_output)
    hidden_output = layers.Dropout(dropout_prob)(hidden_output)
    output = layers.Dense(1, activation='sigmoid')(hidden_output)
    model = tf.keras.Model(inputs=[input_word_ids], outputs=output)
    return model


def create_vanilla_bert_model(num_tokens: int, nn_hidden_dim: int, dropout_prob: float):
    # define the encoder for bert model
    bert_encoder = TFBertModel.from_pretrained('bert-base-uncased')

    input_word_ids = tf.keras.Input(shape=(num_tokens,), dtype=tf.int32, name="input_word_ids")
    bert_embedding = bert_encoder([input_word_ids])
    bert_cls_tokens = layers.Lambda(lambda seq: seq[:, 0, :])(bert_embedding[0])
    hidden_output = layers.Dense(nn_hidden_dim, activation='relu')(bert_cls_tokens)
    hidden_output = layers.Dropout(dropout_prob)(hidden_output)
    output = layers.Dense(1, activation='sigmoid')(hidden_output)
    model = tf.keras.Model(inputs=[input_word_ids], outputs=output)
    return model
