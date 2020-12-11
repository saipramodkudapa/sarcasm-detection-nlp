import os
import json

from tensorflow.keras import models


def load_pretrained_model(serialization_dir: str) -> models.Model:
    """
    Given serialization directory, returns: model loaded with the pretrained weights.
    """

    # Load Config
    config_path = os.path.join(serialization_dir, "config.json")
    model_path = os.path.join(serialization_dir, "model.ckpt.index")

    model_files_present = all([os.path.exists(path)
                               for path in [config_path, model_path]])
    if not model_files_present:
        raise Exception(f"Model files in serialization_dir ({serialization_dir}) "
                        f" are missing. Cannot load_the_model.")

    model_path = model_path.replace(".index", "")

    with open(config_path, "r") as file:
        config = json.load(file)

    # Load Model based on model type
    model_name = config.pop("type")
    if model_name == "CNN":
        # To prevent circular imports
        from model import onlyCNNmodel
        model = onlyCNNmodel(**config)
        model.load_weights(model_path)
    elif model_name == "CNN_BiGRU":
        from model import CNNandAttentiveBiGRUmodel
        model = CNNandAttentiveBiGRUmodel(**config)
        model.load_weights(model_path)
    elif model_name == "BERT_CNN":
        from model import create_bert_cnn_model
        model = create_bert_cnn_model(**config)
        model.load_weights(model_path)
    elif model_name == "BERT":
        from model import create_vanilla_bert_model
        model = create_vanilla_bert_model(**config)
        model.load_weights(model_path)
    else:
        from model import onlyCNNmodel
        model = onlyCNNmodel(**config)
        model.load_weights(model_path)
    return model
