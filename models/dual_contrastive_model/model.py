import torch
import torch.nn as nn

class Transformer(nn.Module):
    """
    Transformer class that is a child of the nn.Module class in PyTorch.
    """

    def __init__(self, base_model, num_classes, method):
        """
        Initialize the Transformer class with a base model, number of classes, and a method.

        Args:
            base_model (nn.Module): The base model from PyTorch.
            num_classes (int): The number of classes for classification.
            method (str): The method used for training the model.
        """
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.method = method
        self.linear = nn.Linear(base_model.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)

        # Set gradients to be calculated for all parameters in the base model
        for param in base_model.parameters():
            param.requires_grad_(True)

    def forward(self, inputs):
        """
        Forward pass for the Transformer model.

        Args:
            inputs (dict): Dictionary containing the input data.

        Returns:
            dict: Dictionary containing the predicted outputs, class features, and label features.
        """
        # Get the raw outputs from the base model
        raw_outputs = self.base_model(**inputs)

        # Get the hidden states from the raw outputs
        hiddens = raw_outputs.last_hidden_state

        # Extract the class features from the hidden states
        cls_feats = hiddens[:, 0, :]

        # Depending on the method, set label features and calculate predicts
        if self.method in ['ce', 'scl']:
            label_feats = None
            predicts = self.linear(self.dropout(cls_feats))
        else:
            label_feats = hiddens[:, 1:self.num_classes+1, :]
            predicts = torch.einsum('bd,bcd->bc', cls_feats, label_feats)

        # Return a dictionary containing the predicted outputs, class features, and label features
        outputs = {
            'predicts': predicts,
            'cls_feats': cls_feats,
            'label_feats': label_feats
        }

        return outputs