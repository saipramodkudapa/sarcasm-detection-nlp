import tensorflow as tf
from tensorflow import math


# compute cross entropy loss
def cross_entropy_loss(logits: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    one_hot_labels = tf.one_hot(labels, logits.shape[1])
    batch_loss = tf.nn.softmax_cross_entropy_with_logits(one_hot_labels, logits)
    loss_value = math.reduce_mean(batch_loss)
    return loss_value
