"""Load and Process olid_datasets
   Olid_dataset : tweets that contain offensive word or meaning and another clean
"""

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

    # Create a dictionary to obtain previous two data to convert them into dataset format
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
