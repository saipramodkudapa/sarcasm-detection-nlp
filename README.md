# sarcasm-detection-nlp


This file is best viewed in Markdown reader (eg. https://jbt.github.io/markdown-editor/)

# Credits

**Credits**: The Model structuring, training, prediction, evaluation follows the same programming paradigm as the assignment/s used in NLP course at Stony Brook University implemented by previous NLP TAs - Harsh Trivedi and Matthew Matero.

# Installation

This project is implemented in python 3.6 and tensorflow 2.3.0. Follow these steps to setup your environment:

1. [Download and install Conda](http://https://conda.io/projects/conda/en/latest/user-guide/install/index.html "Download and install Conda")
2. Create a Conda environment with Python 3.6
```
conda create -n nlp-final python=3.6
```

3. Activate the Conda environment. You will need to activate the Conda environment in each terminal in which you want to use this code.
```
conda activate nlp-final
```
4. Install the requirements:
```
pip install -r requirements.txt
```

5. Download spacy model
```
python -m spacy download en_core_web_sm
```

6. Download glove wordvectors:
```
./download_glove.sh
```

# Data

News headlines dataset used for sarcasm classification is stored in `data/` directory. Training, validation, test data in the form of jsonl files. Training can be found in `data/train.jsonl`, validation is in `data/validate.jsonl` and test is in `data/test.jsonl`.

Sarcasm Detection using Hybrid Neural Network
Rishabh Misra, Prahal Arora
Arxiv, August 2019
[Dataset Citation](https://scholar.google.com/citations?view_op=list_works&hl=en&user=EN3OcMsAAAAJ#d=gs_md_cita-d&u=%2Fcitations%3Fview_op%3Dview_citation%26hl%3Den%26user%3DEN3OcMsAAAAJ%26citation_for_view%3DEN3OcMsAAAAJ%3AqjMakFHDy7sC%26tzom%3D420 "Dataset Citation")

Each record consists of three attributes:

is_sarcastic: 1 if the record is sarcastic otherwise 0

headline: the headline of the news article

article_link: link to the original news article. Useful in collecting supplementary data


# Code Overview

## Data Reading File

Code dealing with reading the dataset, generating and managing vocabulary, indexing the dataset to tensorizing it and loading embedding files is present in `data.py`. 
Code related to tokenizing training/validation instances with BERT tokenizer is also added as a requirement for our advanced models.

## Modeling Files

- model.py

There are 4 classifiers that can be instantiated using (`model.py`).
- onlyCNNmodel
- CNNandAttentiveBiGRUmodel
- create_bert_cnn_model
- create_vanilla_bert_model

The following scripts help you operate on these models: `train.py`, `bert.py`, `predict.py`, `evaluate.py`. To understand how to use them, simply run `python train.py -h`, `python predict.py -h`.

### Train:

The script `train.py` lets you train the models on pre-trained word embeddings, while `bert.py` lets you fine tune pre-trained bert representations for our target task. The first two arguments need to be path to the training set and the validation set. 
Using `--model-choice` argument, we can train the desired model for a particular script.


The following command trains the `cnn_gru` model using `train.py` script:

```
python train.py data/train.jsonl \
                  data/validate.jsonl \
                  --model-choice cnn_gru \
                  --gru-output-dim 128 \
                  --nn-hidden-dim 100 \
                  --dropout-prob 0.2 \
                  --embedding-dim 100 \
                  --batch-size 32 \
                  --num-epochs 5 \
                  --experiment-name _cnn_attn_bigru \
                  --pretrained-embedding-file data/glove.6B.100d.txt
```

The output of this training is stored in its serialization directory. To prevent clashes, model directory name can be adjusted using `experiment-name` argument. The training script automatically generates serialization directory at the path `"serialization_dirs/{model_name}_{experiment_name}"`. So in this case, the serialization directory is `serialization_dirs/_cnn_attn_bigru`.

Similarly, to train bert based model, use following command:

```
python bert.py data/train.jsonl \
                  data/validate.jsonl \
                  --model-choice bert_cnn \
                  --pretrained-bert-model bert-base-uncased \
                  --nn-hidden-dim 100 \
                  --dropout-prob 0.2 \
                  --batch-size 32 \
                  --num-epochs 3 \
                  --experiment-name _bert_with_cnn \
```

### Predict:

Once a model is trained, you can use its serialization directory and a dataset to make predictions on it. For example, the following command:

```
python predict.py serialization_dirs/_cnn_attn_bigru \
                  data/test.jsonl \
                  --predictions-file predictions/cnn_gru.txt
```
Above command makes a prediction on `data/test.jsonl` using trained model at `serialization_dirs/_cnn_attn_bigru` and stores the predicted labels in `predictions/cnn_gru.txt`.

### Evaluate:

Once the predictions are generated you can evaluate the accuracy and F1 score by passing the original dataset with gold labels and the predictions. For example:

```
python evaluate.py data/test.jsonl predictions/cnn_gru.txt
```


### Software Requirements:

All the libraries required to run the code are mentioned in requirements.txt which can be used to setup the runnable enviroment. 

NOTE - Fine tuning the pretrained bert models may require GPU. Average time per epoch on CPU may take longer than 25 minutes depending on the system hardware. On Google colab - GPU, per epoch time was ~ 100 seconds

- tensorflow V2.3.0
- Spacy
- nltk
