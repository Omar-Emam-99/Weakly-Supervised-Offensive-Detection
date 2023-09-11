import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from models.noiseAsareLoss import (
    NormalizedCrossEntropy,
    ReverseCrossEntropy,
    NCEandRCE,
    MeanAbsoluteError,
    NCEandMAE
)


#Create CustomTrainer that inherit from Trainer Class and overwrite compute_loss with NoiseAware loss function
class CustomTrainer(Trainer):
    """
    A custom PyTorch trainer class that computes the loss using the NCEandMAE module.
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes the loss for the given inputs and model.

        Args:
        - model (nn.Module): The PyTorch model being trained
        - inputs (Dict[str, Tensor]): A dictionary containing the inputs to the model
        - return_outputs (bool): Whether or not to also return the model outputs

        Returns:
        - loss (Tensor): A tensor representing the total loss of the model
        - outputs (Dict[str, Tensor]): A dictionary containing the outputs of the model, if return_outputs is True
        """
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss 
        loss_fct = NCEandMAE(1,1,2)
        #loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        loss = loss_fct(logits , labels)
        return (loss, outputs) if return_outputs else loss


class ClassifierModel:
    """
    ClassifierModel is a class that represents a text classification model using Hugging Face's transformers library.

    Attributes:
    - tokenizer: a tokenizer object used to convert input text to tokens
    - model: a model object used to classify the tokens

    Methods:
    - tokenize(batch): tokenizes input text and returns a dictionary of tokenized inputs
    - compute_metrics(pred): computes classification metrics based on predicted and actual labels
    - train(train_data, eval_data, output_dir_name, num_epochs, evaluation_strategy, batch_size, learning_rate, weight_decay, disable_tqdm, log_level): trains the model on training data and saves the trained model to disk
    - test(test_data): tests the trained model on test data and returns classification metrics

    """

    def __init__(self, model_ckpt: str = "distilbert-base-uncased"):
        """
        Initializes a new instance of the ClassifierModel class.

        Parameters:
        - model_ckpt: a string representing the name of the pre-trained model checkpoint to use

        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        return self.tokenizer(batch["tweets"], padding=True, truncation=True)

    @staticmethod
    def compute_metrics(pred) -> dict:
        """
        Computes classification metrics based on predicted and actual labels.

        Parameters:
        - pred: a tuple containing the predicted labels and actual labels

        Returns:
        - a dictionary containing the classification metrics

        """
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        acc = accuracy_score(labels, preds)
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def train(
        self,
        train_data,
        eval_data,
        output_dir_name: str = "distelbert-for-text-classifiaction",
        num_epochs: int = 3,
        evaluation_strategy: str = "epoch",
        batch_size: int = 64,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.001,
        disable_tqdm: bool = False,
        log_level: str = "error"
    ):
        """
        Trains the model on training data and saves the trained model to disk.

        Parameters:
        - train_data: a dataset object representing the training data
        - eval_data: a dataset object representing the evaluation data
        - output_dir_name: a string representing the name of the directory to save the trained model to
        - num_epochs: an integer representing the number of training epochs to run
        - evaluation_strategy: a string representing the evaluation strategy to use (e.g., "epoch")
        - batch_size: an integer representing the batch size to use during training
        - learning_rate: a float representing the learning rate to use during training
        - weight_decay: a float representing the weight decay to use during training
        - disable_tqdm: a boolean indicating whether to disable the progress bar during training
        - log_level: a string representing the logging level to use during training

        """

        train_encoded = train_data.map(self.tokenize, batched=True, batch_size=None)
        eval_encoded = eval_data.map(self.tokenize, batched=True, batch_size=None)

        logging_steps = len(train_encoded["input_ids"]) // batch_size

        training_args = TrainingArguments(
            output_dir="DBERT",
            num_train_epochs=num_epochs,
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            weight_decay=weight_decay,
            evaluation_strategy=evaluation_strategy,
            disable_tqdm=False,
            logging_steps=logging_steps,
            push_to_hub=False,
            report_to="none",
            log_level=log_level
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            compute_metrics=ClassifierModel.compute_metrics,
            train_dataset=train_encoded,
            eval_dataset=eval_encoded,
            tokenizer=self.tokenizer
        )
            
        trainer.train()

        if os.path.isdir("artifacts"):
            self.model.save_pretrained("artifacts/")
        else:
            os.mkdir("artifacts")
            self.model.save_pretrained("artifacts/")


    def test(self, test_data):
        """
        Tests the trained model on test data and returns classification metrics.

        Parameters:
        - test_data: a dataset object representing the test data

        Returns:
        - a dictionary containing the classification metrics

        """
        test_encoded = test_data.map(self.tokenize, batched=True, batch_size=None)
        
        trainer = Trainer(
            self.model,
            compute_metrics=ClassifierModel.compute_metrics
        )
            

        preds = Trainer.predict(test_encoded)

        return preds.metrics