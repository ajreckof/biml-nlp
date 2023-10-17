#%%
import torch
import utils
from torchtext.vocab import build_vocab_from_iterator
import torch.nn.functional as F

#%%
phrases, emotions = utils.load_file("dataset/train.txt")
# %%
vocab = build_vocab_from_iterator([[word for word in phrase] for phrase in phrases], specials=["<unk>"])
# vocab.get_stoi()[phrases[0][1]]


#%%
stoi = vocab.get_stoi()
max = 0
for phrase in phrases :
    if len(phrase) > max:
        max = len(phrase)
        
print([stoi[word] for word in phrases[0]])
label_encoded_phrases = []
i = 0
for phrase in phrases :
    i+= 1
    if i % 1000 : 
        print(i)
    label_encoded_phrases.append([stoi[word] for word in phrase] + [stoi["<unk>"]] * (max - len(phrase)))
print(len(label_encoded_phrases))
label_encoded_phrases = torch.tensor(label_encoded_phrases)
print(label_encoded_phrases)

# %%

print(F.one_hot(label_encoded_phrases[0], len(vocab))[1][138])
print(F.one_hot(label_encoded_phrases[0], len(vocab))[3][685])
[stoi[word] for word in phrases[0]]

# %%

print(torch.tensor(phrases[0].split(' ').split('')))


# %%
array = []
array.append([1,2,3])
print(array)
# %%
