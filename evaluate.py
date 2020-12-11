#pylint: disable = invalid-name
# inbuilt lib imports:
import json
import argparse
from sklearn.metrics import f1_score


def evaluate(gold_data_path: str, prediction_data_path: str):
    """
    Evaluates accuracy and F1 score of label predictions in ``prediction_data_path``
    based on gold labels in ``gold_data_path``.
    """
    with open(gold_data_path) as file:
        gold_labels = [int(json.loads(line.strip())["label"])
                       for line in file.readlines() if line.strip()]

    with open(prediction_data_path) as file:
        predicted_labels = [int(line.strip())
                            for line in file.readlines() if line.strip()]

    if len(gold_labels) != len(predicted_labels):
        raise Exception("Number of lines in labels and predictions files don't match.")

    correct_count = sum([1.0 if predicted_label == gold_label else 0.0
                         for predicted_label, gold_label in zip(predicted_labels, gold_labels)])

    f1 = f1_score(gold_labels, predicted_labels)
    macro_avg_f1 = f1_score(gold_labels, predicted_labels, average='macro')
    total_count = len(predicted_labels)
    _accuracy = correct_count / total_count
    return _accuracy, f1, macro_avg_f1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate classification predictions.')
    parser.add_argument('gold_data_path', type=str, help='gold data file path.')
    parser.add_argument('prediction_data_path', type=str,
                        help='predictions data file path.')

    args = parser.parse_args()
    accuracy, f1, avg_f1 = evaluate(args.gold_data_path, args.prediction_data_path)
    print(f"Accuracy: {round(accuracy, 2)}")
    print(f"F1 score for sarcastic class: {round(f1, 2)}")
    print(f"Average F1 score: {round(avg_f1, 2)}")
