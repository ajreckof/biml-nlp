import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from multiprocessing import Pool
import signal
from tqdm import tqdm
from functools import partial
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import uuid

from IPython.display import clear_output, display
from inspect import cleandoc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, matthews_corrcoef

class printer(str):
    def __repr__(self):
       return cleandoc(self)
    def __print__(self):
       return cleandoc(self)


class RNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, batch_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        
        
        self.i2e = nn.Linear(input_size, embed_size, dtype=torch.double)

        self.i2s = nn.Linear(embed_size + hidden_size, 1, dtype=torch.double)
        self.i2o = nn.Linear(embed_size + hidden_size, output_size, dtype=torch.double)
        self.i2h = nn.Linear(embed_size + hidden_size, hidden_size, dtype=torch.double)

        nn.init.xavier_uniform_(self.i2e.weight)
        nn.init.xavier_uniform_(self.i2s.weight)
        nn.init.xavier_uniform_(self.i2o.weight)
        nn.init.xavier_uniform_(self.i2h.weight)


    def forward(self,input, is_secondary = False, return_both = False):

        # Get the number of columns
        batch_size, num_columns = input.shape
        
        hidden = self.initHidden(batch_size)
        # Iterate over columns
        for i in range(num_columns - 1):
            batch_word = input[:, i]
            hidden = self.forward_word(batch_word, hidden)
        batch_word = input[:, num_columns-1]
        if return_both :
            return self.forward_word(batch_word, hidden, is_final = True, return_both = True)
        else :    
            return self.forward_word(batch_word, hidden, is_final = True, is_secondary = is_secondary)

    def forward_word(self, input, hidden, is_final = False, is_secondary = False, return_both = False):

        input = F.one_hot(input, num_classes=self.input_size).to(torch.double)

        input = self.i2e(input)
        combined = torch.cat((input, hidden), 1)
        if is_final:
            if return_both:
                return self.i2o(combined), self.i2s(combined)
            elif is_secondary:
                return self.i2s(combined)
            else:
                return self.i2o(combined)
        else :
            return self.i2h(combined)

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size, dtype=torch.double)
    
    def test(self, phrases, emotions, negatives):
        emotions_output, negatives_output = self(phrases, return_both = True)
        emotions_predictions = torch.argmax(emotions_output, dim=1)
        return (
            float(torch.sum(emotions_predictions == emotions))/len(phrases),
            float(torch.sum((negatives_output > 0) == negatives))/len(phrases),
            matthews_corrcoef(emotions, emotions_predictions),
        )

    def confusion_matrix(self, dataset):
        phrases, emotions, _ = dataset.tensors
        return confusion_matrix(emotions, torch.argmax(self(phrases), dim=1))


def plot(train, test, file_name = ""):
    "trace courbe de somme des rec et moyenne glissante par episodes"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(visible= True, which='both')
    ax.set_ylim(0,1)
    plt.plot(np.arange(len(train)), train, label=('train','train_secondary','train_phi'))
    plt.plot(np.arange(len(test)), test, label=('test','test_secondary', 'test_phi'))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    if file_name :
        plt.savefig(f'plot/{file_name}.pdf')


def train_rnn(train_dataset, test_dataset, size_vocab, batch_size=8, nb_epochs=20, lr= 10**-4, secondary_proportion = 0.1, embed_size = 100, hidden_size = 100, with_emotions_weight = True):
    _,emotions, negatives = train_dataset.tensors
    weight_negatives = len(negatives)/torch.sum(negatives) - 1
    emotions_count = torch.bincount(emotions).to(torch.double)
    weight_emotions = len(emotions)/len(emotions_count)/emotions_count
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    rnn = RNN(input_size=size_vocab, embed_size=embed_size, hidden_size=hidden_size, output_size=len(weight_emotions), batch_size=batch_size)
    if with_emotions_weight :
        loss_function = torch.nn.CrossEntropyLoss(weight=weight_emotions)
    else : 
        loss_function = torch.nn.CrossEntropyLoss()
    secondary_loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight_negatives]))
    optimizer = torch.optim.AdamW(rnn.parameters(),lr=lr)
    acc_train = []
    acc_test = []
    acc_secondary_train = []
    acc_secondary_test = []
    phi_train = []
    phi_test = []
    id = str(uuid.uuid4())
    dh = display(printer("epoch : 0"), display_id=id)
    
    for i in range(nb_epochs):
        for sentences, emotions, negatives in tqdm(dataloader):
            optimizer.zero_grad()
            y, ys = rnn(sentences, return_both = True)
            loss = (1-secondary_proportion) * loss_function(y, emotions) + secondary_proportion * secondary_loss_function(ys, negatives) 
            loss.backward()
            optimizer.step()
        
        with torch.no_grad() :
            acc, acc_secondary, phi = rnn.test(*train_dataset.tensors)
            acc_train.append(acc)
            acc_secondary_train.append(acc_secondary)
            phi_train.append(phi)
            
            acc, acc_secondary, phi = rnn.test(*test_dataset.tensors)
            acc_test.append(acc)
            acc_secondary_test.append(acc_secondary)
            phi_test.append(phi)

        if dh != None :
            dh.update(printer(
                f""" 
                    epoch : {i+1}
                    Train : {acc_train[-1]}
                    Test : {acc_test[-1]} 
                """
            ))   
        else : 
            print(printer(
                f""" 
                    epoch : {i+1}
                    Train : {acc_train[-1]}
                    Test : {acc_test[-1]} 
                """
            )) 
    cm_train = rnn.confusion_matrix(train_dataset)
    cm_test = rnn.confusion_matrix(test_dataset)
    clear_output(wait= True)
    return (acc_test, acc_train), (acc_secondary_test, acc_secondary_train), (phi_test, phi_train), (cm_test, cm_train)

def one_argument(func,arg):
	return func(*arg)


        
def test_rnn_with_or_without_emotions_weights(train_dataset, test_dataset, size_vocab, cases, n= 10, num_workers = 1, **kwargs):

    acc = {}
    acc_sec = {}
    phi = {}
    cm = {}
    
    
    for name, case in cases.items() :
        case.update(kwargs)
        train_rnn_one_argument_with_kwargs = partial(one_argument, partial(train_rnn, **case))
        if num_workers > 1 :
            pool = Pool(num_workers)
            def handle_interrupt(signal, frame):
                pool.terminate()  # Terminate the pool of worker processes
                pool.join()  # Wait for the pool to clean up
                print("Main process interrupted. Worker processes terminated.")
                exit(1)

            # Register the signal handler for interrupt signals
            signal.signal(signal.SIGINT, handle_interrupt)
            acc[name], acc_sec[name], phi[name], cm[name] = zip(*
                tqdm(
                    pool.imap_unordered(
                        train_rnn_one_argument_with_kwargs,
                        [(train_dataset, test_dataset, size_vocab)] * n,
                    )
                )
            )

        else : 
            acc[name] = []
            acc_sec[name] = []
            phi[name] = []
            cm[name] = []
                        
            for _ in range(n):
                acc_one, acc_sec_one, phi_one, cm_one = train_rnn_one_argument_with_kwargs((train_dataset, test_dataset, size_vocab))
                acc[name].append(acc_one)
                acc_sec[name].append(acc_sec_one)
                phi[name].append(phi_one)
                cm[name].append(cm_one)

    return  acc, acc_sec, phi, cm
#%%