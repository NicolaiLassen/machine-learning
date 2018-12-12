from __future__ import print_function

import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
import pandas as pd
import time as t
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# Run as GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Text preprocessing
"""

# Import text to summarize
# path ./data
df_content = pd.read_csv("./data/data.txt", header=None, delimiter="\t")

# Tokenize
print("Tokenizing text corpus")
tokens = []
for line in df_content[0]:
    for word in word_tokenize(line):
        if word not in tokens:
            tokens.append(word)

# Dictionary of ix
word_to_ix = {word: i for i, word in enumerate(tokens)}
ix_to_word = {i: word for i, word in enumerate(tokens)}

"""
Model
"""


class TextLearner(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(TextLearner, self).__init__()
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(in_dim, hidden_dim)

        self.lstm = nn.GRU(hidden_dim, hidden_dim)

        self.lin_out = nn.Linear(hidden_dim, out_dim)

        self.soft = nn.LogSoftmax(dim=1)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_dim)

    def forward(self, x):
        out = self.embed(x).view(len(x), 1, -1)
        out, hidden = self.lstm(out, self.hidden)
        out = self.lin_out(out.view(len(x), -1))
        out = self.soft(out)
        return out


# Create tensor from text
def tensor_from_text(text):
    ix = [word_to_ix[word] for word in text]
    return torch.tensor(ix, dtype=torch.long)


# Print predictions
def print_texts(pred):
    words = []
    for index in pred:
        words.append(ix_to_word[index])
    print(' '.join(words))


# DIMS
HIDDEN_DIM = 1000
in_dim = len(tokens)
out_dim = len(tokens)

# Init the model
model = TextLearner(in_dim, HIDDEN_DIM, out_dim)

# NLL loss
criterion = nn.NLLLoss()

# Fixed leaning rate
learning_rate = 0.1

# Stochastic gradient descent for optimisation
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Create inputs and target tensor
input_tensor = tensor_from_text(tokens)
labels = tensor_from_text(tokens)

print("Traning the model")

iter = 0
for epoch in range(100):  # Fix number of epoch

    # Zero out gradient
    model.zero_grad()

    # Init the hidden state
    hidden = model.init_hidden()

    # Clear gradients w.r.t. parameters
    optimizer.zero_grad()

    # Forward pass to get logits
    outputs = model(input_tensor)

    # Calculate Loss: softmax --> NLL Loss
    loss = criterion(outputs, labels)

    # Get predictions from the maximum value
    _, predicted = torch.max(outputs.data, 1)

    # Getting gradients w.r.t. parameters
    loss.backward()

    # Updating parameters
    optimizer.step()

    # Print values for each iteration
    correct = (predicted.numpy() == labels.numpy()).sum()
    accuracy = 100 * correct / len(tokens)
    print("Accuracy: " + str(int(accuracy)) + "%")
    print("Loss: " + str(loss.item()))
    print("Iterations: " + str(iter))

    if accuracy == 100:
        print("Done with 100% accuracy")
        print_texts(predicted.numpy())
        break

    if iter % 5 == 0:
        print_texts(predicted.numpy())

    iter += 1