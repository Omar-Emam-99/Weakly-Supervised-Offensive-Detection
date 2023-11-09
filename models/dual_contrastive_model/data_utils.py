import os
import json
import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    """
    Custom Dataset for loading the data
    """

    def __init__(self, raw_data, label_dict, tokenizer, model_name, method):
        """
        raw_data : list of dict
            The raw data to be loaded
        label_dict : dict
            Dictionary that maps labels to integers
        tokenizer : object
            Tokenizer object to tokenize the data
        model_name : str
            Name of the model
        method : str
            Method used for tokenization
        """
        # List of labels for classification or empty list
        label_list = list(label_dict.keys()) if method not in ['ce', 'scl'] else []
        # Different separator tokens for different models
        sep_token = ['[SEP]'] if model_name == 'bert' else ['</s>']
        dataset = list()
        # Iterate through raw data and append to the dataset
        for data in raw_data:
            tokens = data['text'].lower().split(' ')
            label_id = label_dict[data['label']]
            dataset.append((label_list + sep_token + tokens, label_id))
        self._dataset = dataset

    def __getitem__(self, index):
        """
        index : int
            Index of the item to be fetched
        """
        return self._dataset[index]

    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self._dataset)

def my_collate(batch, tokenizer, method, num_classes):
    """
    Function to collate the data
    """
    # Unzip the batch into separate lists
    tokens, label_ids = map(list, zip(*batch))
    # Tokenize the text
    text_ids = tokenizer(tokens,
                         padding=True,
                         truncation=True,
                         max_length=256,
                         is_split_into_words=True,
                         add_special_tokens=True,
                         return_tensors='pt')

    # If the method is not 'ce' or 'scl', add position ids
    if method not in ['ce', 'scl']:
        positions = torch.zeros_like(text_ids['input_ids'])
        positions[:, num_classes:] = torch.arange(0, text_ids['input_ids'].size(1)-num_classes)
        text_ids['position_ids'] = positions
        
    return text_ids, torch.tensor(label_ids)

def load_data(data, tokenizer, batch_size, model_name, method, workers):
    """
    Function to load the data
    """
    # Define the label dictionary
    label_dict = {'NOT': 0, 'OFF': 1}

    # Initialize the custom dataset
    dataset = MyDataset(data, label_dict, tokenizer, model_name, method)
    # Define the collate function
    collate_fn = partial(my_collate,
                         tokenizer=tokenizer,
                         method=method,
                         num_classes=len(label_dict))
    # Initialize the dataloader
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn, pin_memory=True)

    return dataloader

def format_dataset(path, split):
    """
    Formats the dataset for a specified split (e.g., "train", "dev", "test").

    Args:
    - path (str): Path to the dataset.
    - split (str): The split of the dataset to format.

    Returns:
    - data (list): List of dictionaries, where each dictionary contains 'text' and 'label' keys.

    Example:
    - format_dataset("dataset", "train") returns a list of dictionaries representing the training dataset.
    """

    # `prepare_data` is a function that loads and preprocesses the dataset
    olid_data = prepare_data(path)

    # Extract tweets and labels from the specified split
    tweets, labels = olid_data[split]["tweets"], olid_data["train"]["labels_name"]

    # Initialize variables
    sample, data = {}, []

    # Create a list of dictionaries where each dictionary represents a data sample
    for t in zip(labels, tweets):
        sample["text"], sample["label"] = t
        data.append(sample)
        sample = {}  # Reset sample for the next iteration

    return data