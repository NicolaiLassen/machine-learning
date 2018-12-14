from __future__ import print_function

import torch
import torch.nn as nn
import pandas as pd
from nltk.tokenize import word_tokenize

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
text = []
for line in df_content[0]:
    for word in word_tokenize(line):
        if word not in tokens:
            tokens.append(word)
    text.append(word_tokenize(line))

print(text)

# Dictionary of ix
word_to_ix = {word: i for i, word in enumerate(tokens)}
ix_to_word = {i: word for i, word in enumerate(tokens)}


"""
Model
"""


class TextLearner(nn.Module):
    def __init__(self, in_dim, hidden_dim, embed_dim, out_dim):
        super(TextLearner, self).__init__()

        # Set hidden dim
        self.hidden_dim = hidden_dim

        # Embedding words
        self.embed = nn.Embedding(in_dim, embed_dim)

        # Gated recurrent unit (GRU) - https://en.wikipedia.org/wiki/Gated_recurrent_unit
        self.gru = nn.GRU(embed_dim, hidden_dim)

        # Out
        self.lin_out = nn.Linear(hidden_dim, out_dim)
        self.soft = nn.LogSoftmax(dim=1)

        # Init hidden
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_dim, device=device)

    def forward(self, x):
        out = self.embed(x).view(len(x), 1, -1)
        out, hidden = self.gru(out, self.hidden)
        out = self.lin_out(out.view(len(x), -1))
        out = self.soft(out)
        return out


# Create tensor from text
def tensor_from_text(text):
    ix = [word_to_ix[word] for word in text]
    return torch.tensor(ix, dtype=torch.long, device=device)


# Print predictions
def print_texts(pred):
    words = []
    for index in pred:
        words.append(ix_to_word[index])
    print(' '.join(words))


# DIMS
HIDDEN_DIM = 100
in_dim = len(tokens)
out_dim = len(tokens)   # Use labels when you have different text
embed_dim = len(tokens)

# Init the model
model = TextLearner(in_dim, HIDDEN_DIM, embed_dim, out_dim)

# NLL loss
criterion = nn.NLLLoss()

# Fixed leaning rate
learning_rate = 0.2

# Stochastic gradient descent for optimisation
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

"""
Training loop
"""

print("Traning the model")
itr = 0
results = [0 for i in range(len(text))]
for epoch in range(200):  # Fix number of epoch
    for line in text:

        input_tensor = tensor_from_text(line)
        labels = tensor_from_text(line)

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
        accuracy = 100 * correct / len(line)
        print("Accuracy: " + str(int(accuracy)) + "%")
        print("Loss: " + str(loss.item()))
        print("Iterations: " + str(itr))
        results[itr % len(text)] = predicted.numpy()

        if itr % 10 == 0:
            print_texts(predicted.numpy())

        itr += 1

print("\nDone")
for pred in results:
    print("Predicted")
    print_texts(pred)
