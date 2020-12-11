from typing import List, Dict
from data import read_instances, save_vocabulary, build_vocabulary, \
                 load_vocabulary, index_instances, generate_batches, load_glove_embeddings, \
                 bert_index_instances
import os
from tensorflow.keras import models, optimizers
from model import onlyCNNmodel, CNNandAttentiveBiGRUmodel
import numpy as np
import tensorflow as tf
from loss import cross_entropy_loss
from tqdm import tqdm
import json
import argparse
from sklearn.metrics import f1_score


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
    best_epoch_F1_score = float("inf")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}")

        total_training_loss = 0
        total_correct_predictions, total_predictions = 0, 0
        generator_tqdm = tqdm(list(zip(train_batches, train_batch_labels)))
        for index, (batch_inputs, batch_labels) in enumerate(generator_tqdm):
            with tf.GradientTape() as tape:
                logits = model(**batch_inputs, training=True)["logits"]
                loss_value = cross_entropy_loss(logits, batch_labels)
                # Regularisation
                regularization_lambda = 1e-4
                parameters = model.trainable_variables
                l2_norm = tf.add_n([ tf.nn.l2_loss(each) for each in parameters ])
                regularization = 2 * regularization_lambda * l2_norm

                loss_value += regularization
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
        total_preds, total_labels = [], []
        generator_tqdm = tqdm(list(zip(validation_batches, validation_batch_labels)))
        for index, (batch_inputs, batch_labels) in enumerate(generator_tqdm):
            logits = model(**batch_inputs, training=False)["logits"]
            loss_value = cross_entropy_loss(logits, batch_labels)
            total_validation_loss += loss_value
            batch_predictions = np.argmax(tf.nn.softmax(logits, axis=-1).numpy(), axis=-1)
            total_correct_predictions += (batch_predictions == batch_labels).sum()
            total_predictions += batch_labels.shape[0]
            total_preds.extend(batch_predictions)
            total_labels.extend(batch_labels)
            description = ("Average validation loss: %.2f Accuracy: %.2f "
                           % (total_validation_loss/(index+1), total_correct_predictions/total_predictions))
            generator_tqdm.set_description(description, refresh=False)
        average_validation_loss = total_validation_loss / len(validation_batches)
        validation_accuracy = total_correct_predictions/total_predictions

        f1 = f1_score(total_labels, total_preds)
        avg_f1 = f1_score(total_labels, total_preds, average='macro')
        print(f"Validation F1 score for Sarcastic: {round(float(f1), 4)}")
        print(f"Validation Avg F1 score: {round(float(avg_f1), 4)}")

        if validation_accuracy > best_epoch_validation_accuracy:
            print("Model with best validation accuracy so far: %.2f. Saving the model."
                  % (validation_accuracy))
            model.save_weights(os.path.join(serialization_dir, f'model.ckpt'))
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
               "best_epoch_validation_loss": float(best_epoch_validation_loss),
               "epoch_validation_F1_score": round(float(f1), 4),
               "epoch_validation_avg_F1_score": round(float(avg_f1), 4)
               }

    print("Best epoch validation accuracy: %.4f, validation loss: %.4f"
          %(best_epoch_validation_accuracy, best_epoch_validation_loss))

    return {"model": model, "metrics": metrics}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Model')

    parser.add_argument('train_data_file_path', type=str, help='training data file path')
    parser.add_argument('validation_data_file_path', type=str, help='validation data file path')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--num-epochs', type=int, default=10, help='max num epochs to train for')
    parser.add_argument('--pretrained-embedding-file', type=str,
                        help='if passed, use glove embeddings to initialize the embedding matrix')

    parser.add_argument('--model-choice', type=str, default='cnn', choices=("cnn", "cnn_gru"),
                        help='Choice of model')

    parser.add_argument('--experiment-name', type=str, default="default",
                        help='optional experiment name which determines where to store the model training outputs.')

    parser.add_argument('--num-tokens', type=int, help='num_tokens ', default=25)
    parser.add_argument('--nn-hidden-dim', type=int, help='hidden_dim of fully connected neural network', default=100)
    parser.add_argument('--embedding-dim', type=int, help='embedding_dim of word embeddings', default=100)
    parser.add_argument('--gru-output-dim', type=int, help='Output dimension of GRU layer', default=100)
    parser.add_argument('--dropout-prob', type=float, help="dropout rate", default=0.2)

    args = parser.parse_args()

    # Setting constants
    MAX_NUM_TOKENS = 25
    VOCAB_SIZE = 10000
    GLOVE_COMMON_WORDS_PATH = os.path.join("data", "glove_common_words.txt")

    print("Reading training instances.")
    train_instances = read_instances(args.train_data_file_path, MAX_NUM_TOKENS)
    print("Reading validation instances.")
    validation_instances = read_instances(args.validation_data_file_path, MAX_NUM_TOKENS)

    with open(GLOVE_COMMON_WORDS_PATH) as file:
        glove_common_words = [line.strip() for line in file.readlines() if line.strip()]

    vocab_token_to_id, vocab_id_to_token = build_vocabulary(train_instances, VOCAB_SIZE, glove_common_words)

    train_instances = index_instances(train_instances, vocab_token_to_id)
    validation_instances = index_instances(validation_instances, vocab_token_to_id)
    config = {
      "vocab_size": min(VOCAB_SIZE, len(vocab_token_to_id)),
      "embedding_dim": args.embedding_dim,
      "filters": [4, 6, 8],
      "out_channels": 100,
      "drop_prob": args.dropout_prob,
      "nn_hidden_dim": args.nn_hidden_dim,
      "num_classes": 2}

    if args.model_choice == "cnn":
        model = onlyCNNmodel(**config)
        config["type"] = "CNN"
    elif args.model_choice == "cnn_gru":
        config["gru_hidden_dim"] = args.gru_output_dim
        model = CNNandAttentiveBiGRUmodel(**config)
        config["type"] = "CNN_BiGRU"

    if args.pretrained_embedding_file:
        embeddings = load_glove_embeddings(args.pretrained_embedding_file, args.embedding_dim, vocab_id_to_token)
        model._embeddings.assign(tf.convert_to_tensor(embeddings))

    optimizer = optimizers.Adam()

    save_serialization_dir = os.path.join("serialization_dirs", args.experiment_name )
    if not os.path.exists(save_serialization_dir):
      os.makedirs(save_serialization_dir)

    training_output = train(model, optimizer, train_instances,
                              validation_instances, args.num_epochs,
                              args.batch_size, save_serialization_dir)
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
    metrics_path = os.path.join(save_serialization_dir, "metrics.txt")
    with open(metrics_path, "w") as file:
      json.dump(metrics, file)

    print(f"\nFinal model stored in serialization directory: {save_serialization_dir}")
