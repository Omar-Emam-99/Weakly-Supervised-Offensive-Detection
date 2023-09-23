import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# APL (Active Passive Loss)
class NormalizedCrossEntropy(torch.nn.Module):
    """
    Normalized Cross Entropy loss function for multi-class classification problems.
    """
    def __init__(self, num_classes, scale=1.0):
        """
        Initializes the NormalizedCrossEntropy class.

        :param num_classes: Number of classes in the classification problem.
        :type num_classes: int
        :param scale: Scaling factor for the loss function. Default is 1.0.
        :type scale: float
        """
        super(NormalizedCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        """
        Computes the Normalized Cross Entropy loss function.

        :param pred: Predictions from the neural network.
        :type pred: torch.Tensor
        :param labels: True labels for the predictions.
        :type labels: torch.Tensor
        :return: Normalized Cross Entropy loss value.
        :rtype: torch.Tensor
        """
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        return self.scale * nce.mean()


class ReverseCrossEntropy(torch.nn.Module):
    """
    Reverse Cross Entropy loss function for multi-class classification problems.
    """
    def __init__(self, num_classes, scale=1.0):
        """
        Initializes the ReverseCrossEntropy class.

        :param num_classes: Number of classes in the classification problem.
        :type num_classes: int
        :param scale: Scaling factor for the loss function. Default is 1.0.
        :type scale: float
        """
        super(ReverseCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        """
        Computes the Reverse Cross Entropy loss function.

        :param pred: Predictions from the neural network.
        :type pred: torch.Tensor
        :param labels: True labels for the predictions.
        :type labels: torch.Tensor
        :return: Reverse Cross Entropy loss value.
        :rtype: torch.Tensor
        """
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * rce.mean()


class NCEandRCE(torch.nn.Module):
    """
    Combined Normalized Cross Entropy and Reverse Cross Entropy loss function for multi-class classification problems.
    """
    def __init__(self, alpha, beta, num_classes):
        """
        Initializes the NCEandRCE class.

        :param alpha: Scaling factor for the Normalized Cross Entropy loss function.
        :type alpha: float
        :param beta: Scaling factor for the Reverse Cross Entropy loss function.
        :type beta: float
        :param num_classes: Number of classes in the classification problem.
        :type num_classes: int
        """
        super(NCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.nce = NormalizedCrossEntropy(scale=alpha, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        """
        Computes the combined Normalized Cross Entropy and Reverse Cross Entropy loss function.

        :param pred: Predictions from the neural network.
        :type pred: torch.Tensor
        :param labels: True labels for the predictions.
        :type labels: torch.Tensor
        :return: Combined loss value.
        :rtype: torch.Tensor
        """
        return self.nce(pred, labels) + self.rce(pred, labels)


class MeanAbsoluteError(torch.nn.Module):
    """
    Mean Absolute Error loss function for multi-class classification problems.
    """
    def __init__(self, num_classes, scale=1.0):
        """
        Initializes the MeanAbsoluteError class.

        :param num_classes: Number of classes in the classification problem.
        :type num_classes: int
        :param scale: Scaling factor for the loss function. Default is 1.0.
        :type scale: float
        """
        super(MeanAbsoluteError, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        """
        Computes the Mean Absolute Error loss function.

        :param pred: Predictions from the neural network.
        :type pred: torch.Tensor
        :param labels: True labels for the predictions.
        :type labels: torch.Tensor
        :return: Mean Absolute Error loss value.
        :rtype: torch.Tensor
        """
        pred = F.softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        mae = 1. - torch.sum(label_one_hot * pred, dim=1)
        return self.scale * mae.mean()


class NCEandMAE(torch.nn.Module):
    """
    A PyTorch neural network module that combines normalized cross-entropy and mean absolute error losses.
    """

    def __init__(self, alpha, beta, num_classes):
        """
        Initializes the NCEandMAE module with the given hyperparameters.

        Args:
        - alpha (float): A scaling factor for the normalized cross-entropy loss
        - beta (float): A scaling factor for the mean absolute error loss
        - num_classes (int): The number of classes in the dataset
        """
        super(NCEandMAE, self).__init__()
        self.num_classes = num_classes
        self.nce = NormalizedCrossEntropy(scale=alpha, num_classes=num_classes)
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        """
        Computes the forward pass of the NCEandMAE module.

        Args:
        - pred (Tensor): A tensor representing the predicted labels
        - labels (Tensor): A tensor representing the true labels

        Returns:
        - loss (Tensor): A tensor representing the combined loss of the NCE and MAE losses
        """
        return self.nce(pred, labels) + self.mae(pred, labels)
