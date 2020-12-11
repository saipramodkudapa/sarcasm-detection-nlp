# internal imports
from data import read_instances, ids_labels_from_instances, bert_index_instances
from model import create_bert_cnn_model, create_vanilla_bert_model

# external imports
import argparse
import os
import tensorflow as tf
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Model')

    parser.add_argument('train_data_file_path', type=str,  default='data/train.jsonl', help='training data file path')
    parser.add_argument('validation_data_file_path', type=str, default='data/validate.jsonl', help='validation data file path')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--num-epochs', type=int, default=4, help='max num epochs to train for')
    parser.add_argument('--pretrained-bert-model', type=str, default='bert-base-uncased',
                        help='if passed, use glove embeddings to initialize the embedding matrix')
    parser.add_argument('--model-choice', type=str, choices=("bert_cnn", "bert"), help='Choice of model')

    parser.add_argument('--experiment-name', type=str, default="only_bert",
                        help='optional experiment name which determines where to store the model training outputs.')

    parser.add_argument('--num-tokens', type=int, help='num_tokens ', default=16)
    parser.add_argument('--nn-hidden-dim', type=int, help='hidden_dim of fully connected neural network', default=100)
    parser.add_argument('--dropout-prob', type=float, help="dropout rate", default=0.2)

    args = parser.parse_args()

    print("Reading training instances.")
    train_instances = read_instances(args.train_data_file_path, args.num_tokens)
    print("Reading validation instances.")
    validation_instances = read_instances(args.validation_data_file_path, args.num_tokens)

    # index tokens based on bert model vocab using huggingface transformers library
    train_instances = bert_index_instances(train_instances)
    validation_instances = bert_index_instances(validation_instances)
    config = {
        "num_tokens": args.num_tokens,
        "nn_hidden_dim": args.nn_hidden_dim,
        "dropout_prob": args.dropout_prob}

    # based on model choice, build config and instantiate model
    if args.model_choice == "bert":
        model = create_vanilla_bert_model(**config)
        config["type"] = "BERT"
    elif args.model_choice == "bert_cnn":
        config.update({"num_filters": 200, "filter_size": 4, "embedding_dim": 768})
        model = create_bert_cnn_model(**config)
        config["type"] = "BERT_CNN"
    else:
        model = create_vanilla_bert_model(**config)
        config["type"] = "BERT"

    # compile model before training, accuracy as metric, and binary_crossentropy loss for optimization
    model.compile(tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    train_ids, train_labels = ids_labels_from_instances(train_instances)
    val_ids, val_labels = ids_labels_from_instances(validation_instances)
    # initiate training process
    model_history = model.fit(x=train_ids, y=train_labels, epochs=args.num_epochs, verbose=1, batch_size=args.batch_size
                              , validation_data=(val_ids, val_labels))

    # save model in serialization_dir under correct experiment name as taken through argument
    save_serialization_dir = os.path.join("serialization_dirs", args.experiment_name)
    if not os.path.exists(save_serialization_dir):
        os.makedirs(save_serialization_dir)

    # save model weights
    model.save_weights(os.path.join(save_serialization_dir, f'model.ckpt'))

    # Save config used to build model
    config_path = os.path.join(save_serialization_dir, "config.json")
    with open(config_path, "w") as file:
      json.dump(config, file)

    print(f"\nFinal model stored in serialization directory: {save_serialization_dir}")