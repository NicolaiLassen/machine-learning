## Articles
# Special letter - https://medium.com/@Aj.Cheng/seq2seq-18a0730d1d77
# LSTM unit - https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714

from __future__ import print_function

import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
import pandas as pd
import time as t
import matplotlib.pyplot as plt

"""
Primitive text summarization Model (sequence to sequence)
Not using pre trained embed (glove, word2vec)
"""

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

# Print Some info about the text
print("Tokens")
print(word_to_ix)
print(len(tokens))

"""
The en- and decoder
"""


# Encoder net
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()

        # Init hidden dim
        self.hidden_dim = hidden_dim

        # Embedding layer
        self.embed = nn.Embedding(input_dim, hidden_dim)

        # Using lstm unit
        self.gru = nn.GRU(hidden_dim * input_dim, hidden_dim)  # Ref top

    def forward(self, x, hidden):
        out = self.embed(x).view(1, 1, -1)
        out, hidden = self.gru(out, hidden)
        return out, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_dim, device=device)

# Decoder net
class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Decoder, self).__init__()

        # Init hidden dim
        self.hidden_dim = hidden_dim

        # Embedding layer
        self.embed = nn.Embedding(output_dim, hidden_dim)

        # Using lstm unit
        self.gru = nn.GRU(hidden_dim, hidden_dim)  # Ref top

        # Non-linearity
        self.relu = nn.ReLU()

        # Linear function (readout)
        self.out = nn.Linear(hidden_dim, output_dim)

        # LogSoft
        self.soft = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        print(x)
        t.sleep(0.1)
        out = self.embed(x).view(1, 1, -1)
        print(out)
        out = self.relu(out)
        out, hidden = self.gru(out, hidden)
        out = self.soft(self.out(out[0]))
        return out, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_dim, device=device)


"""                   
Indexing list       
"""


def ix_from_text(text):
    return [word_to_ix[word] for word in text]


def tensor_from_text(text):
    ix = ix_from_text(text)
    return torch.tensor(ix, dtype=torch.long, device=device).view(-1, 1)


"""
Train functions
"""

SUMMERY_LEN = 100

def train(input_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=SUMMERY_LEN):
    # Zero out loss
    loss = 0

    encoder_hidden = encoder.init_hidden()

    # Clear gradients w.r.t. parameters
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_out, encoder_hidden = encoder(input_tensor, encoder_hidden)

    decoder_out, decoder_hidden = decoder(encoder_hidden, input_tensor)

    # loss = criterion(encoder_out, input_tensor)

    # Getting gradients w.r.t. parameters
    # loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss


def trainer(encoder, decoder, n_epochs, learning_rate=0.1):
    # Stochastic gradient descent for optimisation (Optim with ADAM)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)

    # LOSS
    criterion = nn.NLLLoss()

    # The text to summarize
    input_tensor = tensor_from_text(tokens)
    
    for i in range(n_epochs):
        loss = train(input_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, SUMMERY_LEN)
        print(loss)


"""
Init model an primitive feed training
"""

# Simple batch size
vocab_size = len(word_to_ix)
BATCH_SIZE = 100
N_ITERS = 1000

num_epochs = N_ITERS / (vocab_size / BATCH_SIZE)
num_epochs = int(num_epochs)

# Fixed leaning rate
LEARNING_RATE = 0.1

# Dims (fixed in embed)
INPUT_DIM = vocab_size
HIDDEN_DIM = 100

encoder = Encoder(INPUT_DIM, HIDDEN_DIM).to(device)
decoder = Decoder(HIDDEN_DIM, SUMMERY_LEN).to(device)

print("Train this bad boy")
trainer(encoder, decoder, num_epochs, LEARNING_RATE)

# TODO: Write REEDME (white paper on approach)
