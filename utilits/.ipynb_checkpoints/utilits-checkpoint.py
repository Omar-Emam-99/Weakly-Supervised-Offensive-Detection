"""Load , Process olid_datasets and split it"""
import os
from datasets import Dataset
import json
from collections import defaultdict


def load_data(path: str) -> dict:
    """
    Load review data (json format).
    Args:
        path (str): Path of the file to be loaded.
    Returns:
        dict: Dictionary containing the loaded data.
    """
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def prepare_data(path: str) -> dict:
    """
    Load and preprocess the data.
    Args:
        path (str): Path of the file to be preprocessed.
    Returns:
        dict: Dictionary containing the preprocessed data.
    """
    # Load train and test data
    train_data = load_data(os.path.join(path, "olid_train.json"))
    test_data = load_data(os.path.join(path, "olid_test.json"))

    # Create a dictionary to obtain previous two data and convert them into dataset format
    dictData = defaultdict(list)
    olied_dataset = {}
    for data, data_name in zip([train_data, test_data], ["train", "test"]):
        dictData.clear()
        for t_value, label_value in zip(data["tweet"].values(), data["subtask_a"].values()):
            dictData["tweets"].append(t_value)
            dictData["labels"].append(1 if label_value =="OFF" else 0)
            dictData["labels_name"].append(label_value)
        olied_dataset[data_name] =  Dataset.from_dict(dictData)

    return olied_dataset


def split_data(training_data, annotated_data_prec: float = 0.2):
    """
    Splits the given training data into a training set and a test set.

    Args:
        training_data (any): the data to split.
        annotated_data_prec (float): the proportion of the data to use as training data.

    Returns:
        tuple: the training set and the test set.
    """
    return training_data.train_test_split(train_size=annotated_data_prec)


def create_dataset_of_label_propagation(text_data, pred_labels):
    """
    Creates a dataset of text data with corresponding predicted labels.

    Args:
        text_data (list): list of text data.
        pred_labels (list): list of predicted labels.

    Returns:
        Dataset: the dataset of text data with corresponding predicted labels.
    """
    return Dataset.from_dict({"tweets": text_data, "labels": pred_labels})


def save_data_json(data, path):
    """
    Saves the given data as a JSON file at the specified path.

    Args:
        data (dict): The data to be saved as JSON.
        path (str): The path where the JSON file will be saved.

    Returns:
        None
    """
    # Open the file at the specified path in write mode
    with open(path, "w") as file_obj:
        # Serialize the data as JSON and write it to the file
        json.dump(data, file_obj)