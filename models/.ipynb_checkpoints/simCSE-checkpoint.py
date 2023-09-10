import logging
from tqdm import tqdm
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor, device
import transformers
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Type, Union

class SimCSE(object):
    """
    A class for embedding sentences, calculating similarities, and retrieving sentences by SimCSE.
    """
    def __init__(self, model_name_or_path: str,
                device: str = None,
                pooler = None):
        """
        Initializes the SimCSE class.

        Args:
            model_name_or_path (str): The name or path of the pre-trained model to be used for sentence encoding.
            device (str): The device to be used for encoding. If `None`, defaults to "cuda" if available, else "cpu".
            pooler (str): The pooling policy to be used for encoding. If `None`, defaults to "cls" for supervised models
            and "cls_before_pooler" for unsupervised models.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if pooler is not None:
            self.pooler = pooler
        elif "unsup" in model_name_or_path:
            logging.info("Use `cls_before_pooler` for unsupervised models. If you want to use other " \
                        "pooling policy, specify `pooler` argument.")
            self.pooler = "cls_before_pooler"
        else:
            self.pooler = "cls"

    def encode(self, sentence: Union[str, List[str]],
                device: str = None,
                return_numpy: bool = False,
                batch_size: int = 64,
                max_length: int = 128) -> Union[ndarray, Tensor]:
        """
        Encodes the input sentence(s) using the pre-trained model.

        Args:
            sentence (Union[str, List[str]]): The sentence or list of sentences to be encoded.
            device (str): The device to be used for encoding. If `None`, defaults to the device specified during initialization.
            return_numpy (bool): If `True`, returns the encoded sentences as a numpy array instead of a tensor.
            batch_size (int): The batch size to be used for encoding.
            max_length (int): The maximum length of the input sentences.

        Returns:
            Union[ndarray, Tensor]: The encoded sentences as a numpy array or tensor.
        """
        target_device = self.device if device is None else device
        self.model = self.model.to(target_device)

        single_sentence = False
        if isinstance(sentence, str):
            sentence = [sentence]
            single_sentence = True

        embedding_list = []
        with torch.no_grad():
            total_batch = len(sentence) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)
            for batch_id in tqdm(range(total_batch)):
                inputs = self.tokenizer(
                    sentence[batch_id*batch_size:(batch_id+1)*batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
                outputs = self.model(**inputs, return_dict=True)
                if self.pooler == "cls":
                    embeddings = outputs.pooler_output
                elif self.pooler == "cls_before_pooler":
                    embeddings = outputs.last_hidden_state[:, 0]
                else:
                    raise NotImplementedError

                embedding_list.append(embeddings.cpu())
        embeddings = torch.cat(embedding_list, 0)

        if single_sentence and not keepdim:
            embeddings = embeddings[0]

        if return_numpy and not isinstance(embeddings, ndarray):
            return embeddings.numpy()

        return embeddings
