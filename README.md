# Weakly-Supervised-Offensive-Detection

## Description 

* This repo discuss a weak-supervision with self-supervised and semi-supervised pseido-labeling with Label-propagation on olid tweets benchmark data sets.

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#dataset">Dataset</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

- advances in self-supervised learning and semi-supervised learning, we combine
them and build a simple yet versatile sentence similarity-based framework for robust
and efficient labeling and data verification.

##### self-supervised learning
- With contrastive learning learn representation that is invariant across two views of the same created via data augmentation , so this approach for images in computer vision we will mimic this idea to suits sentences .
- SimCSE : Model for training contrastive encoders to learn high-quality sentence embeddings.
It aims to maximize the similarity between representations of the same sentence while minimizing the similarity between representations of different sentences.
This allows the model to capture meaningful semantic information in the embeddings.
- Based on the metric from SimCSE we can build a `Nearest Neighbor Graph`.

##### Label Propagation
- As nearby nodes are likely to have the same label, perform label propagation on the nearest neighbor graph to propagate information from samples with known labels to samples without label or with noisy labels .

##### Traning A Classifier Model with new labeled data
- How much annotated labels need to get a reasonable accuray around the one with annotated .
- Use distelbert-base-uncased as an encoder and CrossEntorpy loss for last hidden state head for specific task.
- Try to perform a NoiseAware loss function .

<!-- Getting Started -->
### Getting Started

- Clone the repo and install all requirements and get started .

<!-- Installation -->
#### Installation

1. clone the repo 
    ```
    git clone https://github.com/Omar-Emam-99/Weakly-Supervised-Offensive-Detection.git
    ```
2. install all requirments :
    ```
    pip install -r requirements.txt 
    ```
<!-- Usage -->
## Usage
* Open CLI and run :
1. for generate labels for unannotated data
    ```
    python main.py --generate \
      --path "path_of_data" \
      --split 0.2 \
      --save_model  "path_to_save" \
      --save_data "path_to_save"
    ```
    - in olid dataset we got a training and test set in json files ,so with training data we will split it to :
      - small set (20%) of training to be an annotated 
      - The rest to be unlabeled data

2. Annotate Data with Large language models :
    ```
    python main.py --annotate_with_llms \
      --data_path_llm "path-to-unlabeled-data" \
      --api_token "LLAMA-API-Token"
    ```

3. Train Classifier Model :
    ```
    python main.py --train\
      --data_path "path"\
      --num_epoch 3\
      --test 
    ```
4. Train DCL model on few data and make it as annotator for new data
    ```
    python main.py --DCL 
      --few_data_DCL_train "path"
      --few_data_DCL_valid "path"
      --inference "unlabeled_data"
    ```
<!-- Acknowledgments -->
## Acknowledgments

- Self-supervised Semi-supervised Learning for Data Labeling and Quality Evaluation.[<a href="https://arxiv.org/abs/2111.10932">Paper</a>]
- SimCSE: Simple Contrastive Learning of Sentence Embeddings [<a href="https://arxiv.org/abs/2104.08821">Paper</a>]