#%%
import torch
import utils
from torchtext.vocab import build_vocab_from_iterator
import torch.nn.functional as F

#%%
phrases, emotions = utils.load_file("dataset/train.txt")
# %%

vocab = build_vocab_from_iterator(phrases, specials=["<unk>"])
vocab.get_stoi()[phrases[0].split(' ')[0]]

F.one_hot(torch.tensor([vocab.get_stoi()[[phrase.split(' ')[0] for phrase in phrases]]]), len(vocab))

# %%
print(torch.tensor(phrases[0].split(' ').split('')))