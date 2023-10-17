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
F.one_hot(torch.tensor([[stoi[word] for word in phrase] + [stoi["<unk>"] for i in range(max - len(phrase))] for phrase in phrases]), len(vocab))

# %%
print(torch.tensor(phrases[0].split(' ').split('')))