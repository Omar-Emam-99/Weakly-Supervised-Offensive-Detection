import sys
from pathlib import Path
src_dir = Path.cwd().parent
sys.path.append(str(src_dir))
from utilits.utilits import *
import torch
import os
from tqdm.auto import tqdm
import evaluate
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from collections import defaultdict
from transformers import get_scheduler, AutoConfig
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from models.NoiseAwareLoss import (
    NormalizedCrossEntropy,
    ReverseCrossEntropy,
    NCEandRCE,
    MeanAbsoluteError,
    NCEandMAE
)

class ClassifierAwareLoss:
    """
    ClassifierAwareLoss is a class that performs training using a sequence classification model and a custom loss function.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, model_ckpt: str = "distilbert-base-uncased"):
        """
        Initializes the ClassifierAwareLoss class.

        Parameters:
        - model_ckpt: The checkpoint name or path for the pre-trained model.

        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=2).to(device)

    def tokenize(self, batch):
        """
        Tokenizes input text and returns a dictionary of tokenized inputs.

        Parameters:
        - batch: a dictionary representing a batch of input data

        Returns:
        - a dictionary containing the tokenized inputs

        """
        return self.tokenizer(batch["tweets"], padding="max_length", truncation=True)

    def prepare_data(self, data):
        """
        Prepares the data for training by tokenizing and creating dataloaders.

        Parameters:
        - data: The dataset for training.

        Returns:
        - train_dataloader: The DataLoader for the training dataset.
        - eval_dataloader: The DataLoader for the evaluation dataset.

        """
        tokenized_dataset = train_data.map(self.tokenize, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(["tweets"])
        tokenized_dataset.set_format('torch')

        train_dataloader = DataLoader(tokenized_dataset["train"], shuffle=True, batch_size=16)
        eval_dataloader = DataLoader(tokenized_dataset["test"], shuffle=True, batch_size=16)

        return train_dataloader, eval_dataloader

    def train(self, dataset, num_epochs: int = 5, learning_rate: float = 2e-5, weight_decay: float = 0.001):
        """
        Trains the model using the provided dataset.

        Parameters:
        - dataset: The dataset for training.
        - num_epochs: The number of training epochs.
        - learning_rate: The learning rate for the optimizer.
        - weight_decay: The weight decay for the optimizer.

        """
        # Get data loaders
        train_dataloader, eval_dataloader = self.prepare_data(dataset)

        num_training_steps = num_epochs * len(train_dataloader)
        progress_bar = tqdm(range(num_training_steps), colour="green")

        # Evaluate Metrics
        metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        LR_Scheduler = get_scheduler(name='linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        self.model.train()
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)

                loss_fct = NCEandMAE(1, 1, 2)
                loss = loss_fct(outputs.logits, batch["labels"])
                loss.backward()
                optimizer.step()
                LR_Scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            self.model.eval()
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = self.model(**batch)

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predictions, references=batch["labels"])

        metric.compute()