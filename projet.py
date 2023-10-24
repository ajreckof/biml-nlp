#%%
import torch
import utils
from torchtext.vocab import build_vocab_from_iterator
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from rnn import RNN
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import copy

#%%
phrases_train, emotions_train = utils.load_file("dataset/train.txt")
phrases_test, emotions_test = utils.load_file("dataset/test.txt")
phrases_val, emotions_val = utils.load_file("dataset/val.txt")


# %%
vocab = build_vocab_from_iterator([[word for word in phrase] for phrase in phrases_train], specials=["<unk>"])

#%%

stats.describe([len(phrase) for phrase in phrases_train])
# longueur moyenne des phrases = 20
# variance = 120
# Longueur de phrase -> 30 mots

#%%

def prepare (phrases, sentence_length, emotions, stoi, labelEncoder):
    print("start preparing data")
    label_encoded_phrases = []
    i = 0
    s_unknown = stoi["<unk>"]
    for phrase in phrases :
        i+= 1
        if i % 1000 == 0:
            print(i)
        label_encoded_phrases.append([stoi[word] if word in stoi else s_unknown  for word in phrase[:sentence_length]] + [s_unknown] * (sentence_length - len(phrase)))

    print("preparing target")
    # Préparation des données des émotions
    label_encoded_emotions = labelEncoder.transform(emotions)

    # Dataset
    return TensorDataset(torch.tensor(label_encoded_phrases), torch.tensor(label_encoded_emotions))

#%%

# Préparation des données
sentence_length = 30

print("before label encoder")
stoi = vocab.get_stoi()
labelEncoder = LabelEncoder()
labelEncoder.fit(emotions_train)
print("before prepare ")
# Dataset d'entrainement
train_dataset = prepare(phrases_train, sentence_length, emotions_train, stoi, labelEncoder)
test_dataset = prepare(phrases_test, sentence_length, emotions_test, stoi, labelEncoder)

#%%

nb_epochs = 100
batch_size = 25

dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

rnn = RNN(input_size=len(vocab), embed_size=20, hidden_size=10, output_size=len(set(emotions_train)), batch_size=batch_size)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(),lr=0.001)

for i in range(nb_epochs):
    print(i)
    for sentences, emotions in dataloader:
        optimizer.zero_grad()
        y = rnn(sentences)
        loss = loss_function(y, emotions)
        loss.backward()
        optimizer.step()
    
    sentences_test, emotions_test = test_dataset.tensors
    correct = torch.argmax(rnn(sentences_test)) == emotions_test
    correct_train_sum = 0
    correct_train_len = 0
    for sentences, emotions in dataloader:
        correct_train = torch.argmax(rnn(sentences)) == emotions
        correct_train_sum += sum(correct_train)
        correct_train_len += len(correct_train)
    print(float(correct_train_sum/correct_train_len))
    print(float(sum(correct)/len(correct)))

        
# %%
