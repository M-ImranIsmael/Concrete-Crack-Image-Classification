# Concrete Crack Image Classification

The Concrete Cracks Image Classification project uses deep learning to categorize images into positive (concrete cracks) and negative (no concrete cracks) categories. Transfer learning is utilized to fine-tune a pre-trained model on a dataset of thousands of images to achieve high classification accuracy. Results include a trained model and visualizations/metrics providing insights into model performance.

# Acknowledgment of Data 💕

The project is made possible with the dataset obtained from:

[Çağlar Fırat Özgenel's Concrete Crack Images for Classification](https://data.mendeley.com/datasets/5y9wdsg2zt/2)

## Build With

<p align="left">
  <a href="https://www.python.org" target="_blank" rel="noreferrer">
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/>
  </a>
  <a href="https://www.tensorflow.org" target="_blank" rel="noreferrer">
    <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/>
  </a>
  <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer">
    <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/>
  </a>
  <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer">
    <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/>
  </a>
  <a href="https://numpy.org/" target="_blank" rel="noreferrer">
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/numpy/numpy-original.svg" alt="numpy" width="40" height="40"/>
  </a>
  <a href="https://matplotlib.org/" target="_blank" rel="noreferrer">
    <img src="https://upload.wikimedia.org/wikipedia/en/5/56/Matplotlib_logo.svg" alt="matplotlib" width="40" height="40"/>
  </a>
  <a href="https://code.visualstudio.com/" target="_blank" rel="noreferrer">
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/9a/Visual_Studio_Code_1.35_icon.svg" alt="vscode" width="40" height="40"/>
  </a>
  <a href="https://opencv.org/" target="_blank" rel="noreferrer"> 
  <img src="https://www.vectorlogo.zone/logos/opencv/opencv-icon.svg" 
  alt="opencv" width="40" height="40"/> 
  </a>
</p>

# Directory structure

- [training.py](): Contains code for loading/preprocessing data, defining/compiling/fitting the transfer learning model.

- [pictures](): Folder containing plots and results, including confusion matrix and training/validation accuracy/loss graphs.

- [saved_models](): Folder containing saved trained model.

- [model.png](): Model architecture for the base model (MobileV2Net)

# Results

## EDA

The plot below shows 9 examples of images and their corresponding labels in this dataset.
![alt text](pictures/Imran_example_positive_negative.png)

## Image Augmentation

This section demonstrates the use of data augmentation to enhance the model's performance and expand the dataset. It showcases images transformed using various techniques such as rotation, and flip.

![alt text](pictures/Imran_image_augmentation.png)

## Model Architecture

The model architecture is based on MobileNetV2 with a global average pooling layer and a dropout layer, followed by a dense layer with 2 output units for binary classification.

Here is a summary of the model architecture:

![alt text](pictures/Imran_model_architecture.png)

## Evaluation Before and After Training

This section shows the evaluation metrics (accuracy and loss) of the model before and after training.

![alt text](pictures/Imran_model_evalutation_before_training.png)
![alt text](pictures/Imran_model_evalutation_after_training.png)

## Tensorboard Result

The two plots below show the training progress of the LSTM model using TensorBoard.

The first plot shows the epoch loss for both the training and validation datasets. The loss is calculated as the difference between the predicted and actual values for each time step, and the lower the loss, the better the model performs.

![alt text](pictures/Imran_tensorboard_epochaloss.png)

The second plot shows the epoch accuracy for both the training and validation datasets. Accuracy is a metric used to evaluate the performance of the model, and it measures the percentage of correct predictions made by the model. The higher the accuracy, the better the model performs.

![alt text](pictures/Imran_tensorboard_epochaccuracy.png)

## Classification Report

![alt text](pictures/Imran_classification_report.png)

## Confusion Matrix

![alt text](pictures/Imran_confusion_matrix.png)

## Model Predictions on Test Data

This section shows some examples of the model's predictions on the test dataset
![alt text](pictures/Imran_predicted_on_testdata.png)

## Predicted vs True Label

This section shows plots of the predicted labels vs. the true labels for the test dataset.
![alt text](pictures/Imran_predicted_vs_true_label.png)
