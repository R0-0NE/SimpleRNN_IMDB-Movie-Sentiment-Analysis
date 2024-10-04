# SimpleRNN_IMDB-Movie-Sentiment-Analysis

### Project Overview
This project aims to perform sentiment analysis on the IMDB movie reviews dataset using a Simple Recurrent Neural Network (RNN). The goal is to classify movie reviews as positive or negative based on their content.

### Dataset
The IMDB Movie Review Dataset consists of 50,000 highly polar movie reviews, split evenly into 25,000 training and 25,000 test samples. The reviews are labeled as either positive or negative, with no neutral reviews.

The dataset can be loaded using the keras.datasets API, which automatically splits it into training and test sets.

### Model Architecture
The model is built using a Simple RNN network. Recurrent Neural Networks (RNNs) are particularly useful for sequential data like text because they can capture dependencies between words across the sequence.

Steps:
1. Embedding Layer: The input words are converted into dense vectors of fixed size using an Embedding layer.
2. Simple RNN Layer: A recurrent layer that processes the input word vectors one at a time, maintaining a memory of past inputs.
3. Dense Output Layer: The RNN output is fed into a fully connected layer with a single neuron and a sigmoid activation function for binary classification (positive/negative).

### Model Summary:
1. Embedding Layer: Converts words into dense vectors.
2. Simple RNN Layer: Processes sequences of word embeddings to learn dependencies.
3. Dense Layer: Outputs a probability that the review is positive.

### Model Training
The model is trained using the binary cross-entropy loss function and the Adam optimizer. The dataset is split into training and validation sets to evaluate the model's performance after each epoch.

We used early stopping to prevent overfitting by monitoring the validation loss.

### Performance
The model achieves an accuracy of approximately 80-85% on the test set, effectively classifying movie reviews as positive or negative.

### Prerequisites
Python 
TensorFlow / Keras
Numpy
Google Colab
streamlit
