import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize

# Run as GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Text preprocessing
"""

# Import text to summarize
# path ./data
df_wine = pd.read_csv("./data/winemag-data-130k-v2.csv")

# Tokenize
print("Tokenizing text corpus")
tokens = []
text = []
label = []
for i in range(100):
    for word in word_tokenize(df_wine["description"][i]):
        if word not in tokens:
            tokens.append(word)
    text.append(word_tokenize(df_wine["description"][i]))
    label.append(df_wine["variety"][i])

print(text)
print(label)

# Dictionary of ix
word_to_ix = {word: i for i, word in enumerate(tokens)}
label_to_ix = {label: i for i, label in enumerate(label)}
ix_to_label = {i: label for i, label in enumerate(label)}


"""
Model
"""

class RNN(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, out_dim):
        super(RNN, self).__init__()

        # Embedding words
        self.embeds = nn.Embedding(input_dim, embed_dim)
        # RNN
        self.rnn = nn.GRU(embed_dim, hidden_dim)
        # Lin out
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = x.transpose(1, 0)
        out = self.embeds(x)
        out, hidden = self.rnn(out)
        out = self.fc(hidden.view(1, -1))
        return out


# Create tensor from text
def tensor_from_text(text):
    return torch.tensor([[word_to_ix[word] for word in text]], dtype=torch.long, device=device)


# Create winery tensor
def tensor_from_label(label):
    return torch.tensor([label_to_ix[label]], dtype=torch.long, device=device)


text_batch = []
label_batch = []

# Create batches
for i in range(len(text)):
    text_batch.append(tensor_from_text(text[i]))
    label_batch.append(tensor_from_label(label[i]))


# DIMS
EMBEDDING_DIM = 100
HIDDEN_DIM = 100
INPUT_DIM = len(tokens)
OUTPUT_DIM = len(label)

# Init the model
model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

# NLL loss
criterion = nn.CrossEntropyLoss()

# Fixed leaning rate
learning_rate = 0.1

# Stochastic gradient descent for optimisation
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

"""
Training loop
"""

loss_array = []
epochs_array = []
loss = 0

print("\nTraning the model")
for epoch in range(10):  # Fix number of epoch

    pred = []
    act = []

    for i in range(len(text_batch)):
        # Create tensors
        input_tensor = text_batch[i]
        label_tensor = label_batch[i]

        # Zero out gradient
        model.zero_grad()

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get logits
        outputs = model(input_tensor)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, label_tensor)

        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        pred.append(predicted[0].item())
        act.append(label_tensor[0].item())
        loss = loss.item()

    # Prints pr epoch
    correct = (np.asarray(pred) == np.asarray(act)).sum()
    accuracy = 100 * correct / len(text)
    print("Accuracy: {}%".format(int(accuracy)))
    print("Loss: {}".format(loss))
    print("Epoch: {}\n".format(epoch+1))

    # Plot
    loss_array.append(loss)
    epochs_array.append(epoch+1)

"""
Testing model
"""

print("Done")
print("Trying to predict text: \"{}\"".format(df_wine["description"][0]))
outputs = model(text_batch[0])
_, predicted = torch.max(outputs.data, 1)
print("Predicted: {}".format(ix_to_label[predicted[0].item()]))
print("Actual: {}".format(ix_to_label[(label_batch[0])[0].item()]))


"""
Plot error
"""

plt.figure(1)
plt.plot(epochs_array, loss_array)
plt.title("RNN loss")
plt.xlabel('Training epoch')
plt.ylabel('Classification loss (%)')
plt.show()
