# Bigram-Language-Model
A character-level language model built from scratch, first with statistics (bigrams) and then with a simple neural network using PyTorch.
## Character-Level Language Model from Scratch

This repository contains my implementation of a character-level language model, inspired by Andrej Karpathy's "makemore" series.

The project is built in a single Jupyter Notebook (`MakeMore.ipynb`) and explores two different approaches to solving the same problem:

### Part 1: Statistical Bigram Model
The first version is a simple language model built on statistics. It works by counting the occurrences of character pairs (bigrams) in a large dataset of names to determine the probability of the next letter.

* Calculates a probability matrix for all character pairs.
* Samples from this matrix to generate new names.
* Visualizes the probabilities using a heatmap.

![Heatmap of Bigram Probabilities](C:\Users\preet\OneDrive\Desktop\heatmap.png)

### Part 2: Neural Network Model
The second part reframes the problem using a simple neural network. This version uses PyTorch to build a model that *learns* the probabilities through gradient descent.

* Uses one-hot encoding to feed character data into the network.
* Builds a basic neural network to predict the next character.
* Calculates the negative log-likelihood loss and uses backpropagation to train the model.
