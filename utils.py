import torch
from torchtext.vocab import build_vocab_from_iterator
import torch.nn.functional as F

def load_file(file):
    f = open(file, 'r')
    # read each line and separate data by ;
    text = []
    emotion = []
    for line in f:
        line_splited = line.replace('\n', '').split(';')
        text.append(line_splited[0].split(' '))
        emotion.append(line_splited[1])
    f.close()
    return text, emotion

