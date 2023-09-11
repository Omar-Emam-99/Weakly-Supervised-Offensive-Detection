from models.LP import LabelPropagation , LabelSpreading
from models.simCSE import SimCSE
import joblib


class DataLabeling :

    def __init__(self,
                 encoder_model_ckpt : str ="princeton-nlp/sup-simcse-roberta-base",
                 kernel: str ='knn',
                 n_jobs: int =-1,
                 n_neighbors : int =20):
        """
        Initializes the DataLabeling class with the specified parameters.

        Parameters:
        encoder_model_ckpt (str): The path to the encoder model checkpoint.
        kernel (str): The kernel used in the LabelSpreading algorithm.
        n_jobs (int): The number of parallel jobs to run for the LabelSpreading algorithm.
        n_neighbors (int): The number of neighbors used in the LabelSpreading algorithm.
        """
        self.simCSE = SimCSE(encoder_model_ckpt)
        self.LP = LabelSpreading(kernel=kernel, n_jobs=n_jobs, n_neighbors=n_neighbors)

    def encode(self,data):
        """
        Encodes the given data using the SimCSE encoder.

        Parameters:
        data: The data to be encoded.

        Returns:
        The encoded data.
        """
        return self.simCSE.encode(data)


    def generate_labels(self,annoteted_data,
                       annotations,
                       unlabeled_data):
        """
        Generates labels for the unlabeled data using the LabelSpreading algorithm.

        Parameters:
        annoteted_data: The annotated data used to fit the LabelSpreading algorithm.
        labels: The labels for the annotated data.
        unlabeled_data: The unlabeled data to generate labels for.

        Returns:
        The predicted labels for the unlabeled data.
        """
        annot_data_embds = self.encode(annoteted_data)
        #Create knn Graph
        self.LP.fit(annot_data_embds.numpy() , annotations)

        unlabeled_data_embds = self.encode(unlabeled_data)

        return self.LP.predict(unlabeled_data_embds)
    
    def save(self, filename: str):
        """
        Save the LP object to a file with the given filename.

        Args:
            filename (str): The name of the file to save the LP object to.
        """
        # Use joblib.dump to save the LP object to the specified file
        joblib.dump(self.LP, filename)        