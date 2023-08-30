# Music-Genre-Classification-with-Deep-Learning

-----------------AUDIO GENRE CLASSIFICATION PROJECT-----------------------

This repository contains the necessary code and resources for an audio genre classification project using Python and TensorFlow. The goal of this project is to build and evaluate various machine learning models for classifying audio clips into different music genres. The dataset consists of 10 audio wave files for each of the following genres:

Rock
Blues
Classical
Country
Disco
Hiphop
Jazz
Metal
Pop
Reggae


----------------FOLDER STRUCTURE----------------------------------------

The repository is organised as follows:

1. dataset

This folder contains the audio wave files for each music genre. There are 10 audio clips available for each genre.

2. json_files

This folder contains JSON files that store the results of model evaluations. These JSON files include accuracy, error, and Area Under the Curve (AUC) values obtained from 100 runs for each model. The available JSON files are:

features.json: JSON file for extracted audio features.

cnn_results.json: Accuracy and error results for CNN model.
cnn_auc.json: AUC values for CNN model.

lstm_results.json: Accuracy and error results for LSTM model.
lstm_auc.json: AUC values for LSTM model.

3. model_images

This folder contains image diagrams of the different machine learning models used in the project. The model diagrams visualize the architecture of each model. The available model images are:

model_mlp.png: MLP model architecture diagram.
model_cnn.png: CNN model architecture diagram.
model_lstm.png: LSTM model architecture diagram.

4. Source

This folder contains the source code files necessary for building, training, and evaluating the machine learning models:


MLP_Model.py: MLP model architecture, training and metrics plot code.
CNN_model.py: CNN model architecture, training and metrics plot code.
LSTM_model.py: LSTM model architecture, training and metrics plot code.

AUC_compare.py: Comparison of AUC values from auc json files.

CNN_stats.py: CNN model statistics and analysis from results json files.
LSTM_stats.py: LSTM model statistics and analysis from results json files.

Features_extraction.py: Audio feature extraction code.


--------------------USAGE-------------------------------------------------

To use this repository, follow these steps:

Ensure you have Python and TensorFlow installed.


Run the model training scripts (CNN_model.py, LSTM_model.py, MLP_Model.py) to train the respective models.

Run the evaluation scripts (AUC_compare.py, CNN_stats.py, LSTM_stats.py) to analyze model performance with the 100 runs that I have done.

Explore the extracted features using Features_extraction.py.

View the model architectures in the model_images folder.

Feel free to modify and adapt the code to suit your specific requirements.
