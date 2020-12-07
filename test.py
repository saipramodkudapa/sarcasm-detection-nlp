from typing import List, Dict
from data import read_instances, save_vocabulary, build_vocabulary, \
                 load_vocabulary, index_instances, generate_batches, load_glove_embeddings
import os
from tensorflow.keras import models, optimizers
from main_model import MainClassifier
import numpy as np
import tensorflow as tf
from loss import cross_entropy_loss
from tqdm import tqdm
import json


def train(model: models.Model,
          optimizer: optimizers.Optimizer,
          train_instances: List[Dict[str, np.ndarray]],
          validation_instances: List[Dict[str, np.ndarray]],
          num_epochs: int,
          batch_size: int,
          serialization_dir: str = None) -> tf.keras.Model:
    """
    Trains a model on the give training instances as configured and stores
    the relevant files in serialization_dir. Returns model and some important metrics.
    """

    print("\nGenerating Training batches:")
    train_batches = generate_batches(train_instances, batch_size)
    print("Generating Validation batches:")
    validation_batches = generate_batches(validation_instances, batch_size)

    train_batch_labels = [batch_inputs.pop("labels") for batch_inputs in train_batches]
    validation_batch_labels = [batch_inputs.pop("labels") for batch_inputs in validation_batches]

    tensorboard_logs_path = os.path.join(serialization_dir, f'tensorboard_logs')
    tensorboard_writer = tf.summary.create_file_writer(tensorboard_logs_path)
    best_epoch_validation_accuracy = float("-inf")
    best_epoch_validation_loss = float("inf")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}")

        total_training_loss = 0
        total_correct_predictions, total_predictions = 0, 0
        generator_tqdm = tqdm(list(zip(train_batches, train_batch_labels)))
        for index, (batch_inputs, batch_labels) in enumerate(generator_tqdm):
            with tf.GradientTape() as tape:
                logits = model(**batch_inputs, training=True)["logits"]
                loss_value = cross_entropy_loss(logits, batch_labels)
                grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            total_training_loss += loss_value
            batch_predictions = np.argmax(tf.nn.softmax(logits, axis=-1).numpy(), axis=-1)
            total_correct_predictions += (batch_predictions == batch_labels).sum()
            total_predictions += batch_labels.shape[0]
            description = ("Average training loss: %.2f Accuracy: %.2f "
                           % (total_training_loss/(index+1), total_correct_predictions/total_predictions))
            generator_tqdm.set_description(description, refresh=False)
        average_training_loss = total_training_loss / len(train_batches)
        training_accuracy = total_correct_predictions/total_predictions

        total_validation_loss = 0
        total_correct_predictions, total_predictions = 0, 0
        generator_tqdm = tqdm(list(zip(validation_batches, validation_batch_labels)))
        for index, (batch_inputs, batch_labels) in enumerate(generator_tqdm):
            logits = model(**batch_inputs, training=False)["logits"]
            loss_value = cross_entropy_loss(logits, batch_labels)
            total_validation_loss += loss_value
            batch_predictions = np.argmax(tf.nn.softmax(logits, axis=-1).numpy(), axis=-1)
            total_correct_predictions += (batch_predictions == batch_labels).sum()
            total_predictions += batch_labels.shape[0]
            description = ("Average validation loss: %.2f Accuracy: %.2f "
                           % (total_validation_loss/(index+1), total_correct_predictions/total_predictions))
            generator_tqdm.set_description(description, refresh=False)
        average_validation_loss = total_validation_loss / len(validation_batches)
        validation_accuracy = total_correct_predictions/total_predictions

        if validation_accuracy > best_epoch_validation_accuracy:
            print("Model with best validation accuracy so far: %.2f. Saving the model."
                  % (validation_accuracy))
            classifier.save_weights(os.path.join(serialization_dir, f'model.ckpt'))
            best_epoch_validation_loss = average_validation_loss
            best_epoch_validation_accuracy = validation_accuracy

        with tensorboard_writer.as_default():
            tf.summary.scalar("loss/training", average_training_loss, step=epoch)
            tf.summary.scalar("loss/validation", average_validation_loss, step=epoch)
            tf.summary.scalar("accuracy/training", training_accuracy, step=epoch)
            tf.summary.scalar("accuracy/validation", validation_accuracy, step=epoch)
        tensorboard_writer.flush()

    metrics = {"training_loss": float(average_training_loss),
               "validation_loss": float(average_validation_loss),
               "training_accuracy": float(training_accuracy),
               "best_epoch_validation_accuracy": float(best_epoch_validation_accuracy),
               "best_epoch_validation_loss": float(best_epoch_validation_loss)}

    print("Best epoch validation accuracy: %.4f, validation loss: %.4f"
          %(best_epoch_validation_accuracy, best_epoch_validation_loss))

    return {"model": model, "metrics": metrics}


train_data_path = 'data/train.jsonl'
val_data_path = 'data/validate.jsonl'
glove_path = 'data/glove.6B.100d.txt'
MAX_NUM_TOKENS = 25
GLOVE_COMMON_WORDS_PATH = os.path.join("data", "glove_common_words.txt")
VOCAB_SIZE = 10000
num_epochs = 10
batch_size = 32

print("Reading training instances.")
train_instances = read_instances(train_data_path, MAX_NUM_TOKENS)
print("Reading validation instances.")
validation_instances = read_instances(val_data_path, MAX_NUM_TOKENS)

with open(GLOVE_COMMON_WORDS_PATH) as file:
    glove_common_words = [line.strip() for line in file.readlines() if line.strip()]
vocab_token_to_id, vocab_id_to_token = build_vocabulary(train_instances, VOCAB_SIZE,
                                                        glove_common_words)

train_instances = index_instances(train_instances, vocab_token_to_id)
validation_instances = index_instances(validation_instances, vocab_token_to_id)

embeddings = load_glove_embeddings(glove_path, 100, vocab_id_to_token)
optimizer = optimizers.Adam()

config = {
    "vocab_size": min(VOCAB_SIZE, len(vocab_token_to_id)),
    "embedding_dim": 100,
    "filters": [4, 6, 8],
    "out_channels": 100,
    "drop_prob": 0.2,
    "hidden_units": 200,
    # "gru_hidden_size": 256,
    "num_classes": 2}

classifier = MainClassifier(**config)
classifier._embeddings.assign(tf.convert_to_tensor(embeddings))

config["type"] = "main"
save_serialization_dir = os.path.join("serialization_dirs", 'main_model' + '_3_only_cnn')
if not os.path.exists(save_serialization_dir):
    os.makedirs(save_serialization_dir)

training_output = train(classifier, optimizer, train_instances,
                            validation_instances, num_epochs,
                            batch_size, save_serialization_dir)

classifier = training_output["model"]
metrics = training_output["metrics"]

# Save the used vocabulary
vocab_path = os.path.join(save_serialization_dir, "vocab.txt")
save_vocabulary(vocab_id_to_token, vocab_path)

# Save the used config
config_path = os.path.join(save_serialization_dir, "config.json")
with open(config_path, "w") as file:
    json.dump(config, file)

# Save the training metrics
metrics_path = os.path.join(save_serialization_dir, "metrics.json")
with open(metrics_path, "w") as file:
    json.dump(metrics, file)

print(f"\nFinal model stored in serialization directory: {save_serialization_dir}")
