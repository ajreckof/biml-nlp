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
from IPython.display import clear_output
import matplotlib.pyplot as plt

#%%
phrases_train, emotions_train = utils.load_file("dataset/train.txt")
phrases_test, emotions_test = utils.load_file("dataset/test.txt")
phrases_val, emotions_val = utils.load_file("dataset/val.txt")

#%%
# Create an empty dictionary to store counts
count_dict = {}

# Iterate through the list and update the counts in the dictionary
for phrase in phrases_train:
    for word in phrase :
        if word in count_dict:
            count_dict[word] += 1
        else:
            count_dict[word] = 1
values = list(count_dict.values())
for i in range(19):
    print((i+1)*5,":" ,stats.scoreatpercentile(values, (i+1)*5))
# %%
vocab = build_vocab_from_iterator([[word for word in phrase] for phrase in phrases_train], specials=["<unk>"], min_freq= 5)

#%%

stats.describe([len(phrase) for phrase in phrases_train])
# longueur moyenne des phrases = 20
# variance = 120
# Longueur de phrase -> 30 mots

#%%

def prepare (phrases, sentence_length, emotions, stoi, labelEncoder):
    label_encoded_phrases = []
    s_unknown = stoi["<unk>"]
    for phrase in phrases :
        label_encoded_phrases.append([stoi[word] if word in stoi else s_unknown  for word in phrase[:sentence_length]] + [s_unknown] * (sentence_length - len(phrase)))

    # Préparation des données des émotions
    label_encoded_emotions = labelEncoder.transform(emotions)

    # Dataset
    return TensorDataset(torch.tensor(label_encoded_phrases), torch.tensor(label_encoded_emotions))

#%%

# Préparation des données
sentence_length = 30

stoi = vocab.get_stoi()
labelEncoder = LabelEncoder()
labelEncoder.fit(emotions_train)
# Dataset d'entrainement
train_dataset = prepare(phrases_train, sentence_length, emotions_train, stoi, labelEncoder)
test_dataset = prepare(phrases_test, sentence_length, emotions_test, stoi, labelEncoder)

#%%

nb_epochs = 30
batch_size = 25

dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

rnn = RNN(input_size=len(vocab), embed_size=20, hidden_size=20, output_size=len(set(emotions_train)), batch_size=batch_size)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(),lr=0.001)
acc_train = []
acc_test = []
for i in range(nb_epochs):
    for sentences, emotions in dataloader:
        optimizer.zero_grad()
        y = rnn(sentences)
        loss = loss_function(y, emotions)
        loss.backward()
        optimizer.step()
    
    
    with torch.no_grad() :
        sentences_test, emotions_test = test_dataset.tensors
        correct = torch.sum(torch.argmax(rnn(sentences_test), dim=1) == emotions_test)
        correct_train_sum = 0
        correct_train_len = 0
        for sentences, emotions in dataloader:
            correct_train = torch.argmax(rnn(sentences), dim=1) == emotions
            correct_train_sum += sum(correct_train)
            correct_train_len += len(correct_train)
        
        acc_train.append(float(correct_train_sum/correct_train_len))
        acc_test.append(float(correct/len(emotions_test)))
        clear_output(wait= True)
        print("\nepoch : ", i,
            "\nTrain : ", acc_train[-1],
            "\nTest : ", acc_test[-1],
        )

        
# %%
def plot(acc_train, acc_test, file_name = ""):
    "trace courbe de somme des rec et moyenne glissante par episodes"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(visible= True, which='both')
    ax.set_ylim(0,1)
    plt.plot(np.arange(len(acc_train)), acc_train, label='train')
    plt.plot(np.arange(len(acc_test)), acc_test, c='r', label='test')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left');
    if file_name :
        plt.savefig(f'plot/{file_name}.pdf')

# %%
plot(acc_train, acc_test)
# %%
