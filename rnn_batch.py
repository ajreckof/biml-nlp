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


from IPython.display import clear_output, display
from inspect import cleandoc
from sklearn.metrics import confusion_matrix, matthews_corrcoef
import copy

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
		return self.forward_word(batch_word, hidden, is_final = True, return_both = return_both, is_secondary= is_secondary)

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
	
	def test(self, phrases, emotions, negatives, batch_size = 200):
		emotions_output, negatives_output = torch.Tensor(), torch.Tensor()
		for i in range(len(phrases)//batch_size):
			emotions_output_batch, negatives_output_batch = self(phrases[i*200:(i+1)*200], return_both = True)
			emotions_output = torch.cat((emotions_output,emotions_output_batch))
			negatives_output = torch.cat((negatives_output,negatives_output_batch))
		emotions_predictions = torch.argmax(emotions_output, dim=1)
		return (
			float(torch.sum(emotions_predictions == emotions))/len(phrases),
			matthews_corrcoef(emotions, emotions_predictions),
			matthews_corrcoef(negatives, negatives_output > 0),
		)

	def confusion_matrix(self, dataset):
		phrases, emotions, _ = dataset.tensors
		return confusion_matrix(emotions, torch.argmax(self(phrases), dim=1), normalize = "true")

def compare_state_dict(sd1,sd2):
	for x in sd1 :
		if not torch.equal(sd1[x],sd2[x]):
			return False
	return True

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


def train_rnn(train_dataset, val_dataset, test_dataset, size_vocab, batch_size=8, nb_epochs=20, lr= 10**-4, secondary_proportion = 0.1, embed_size = 100, hidden_size = 100, with_emotions_weight = True):
	_,emotions, negatives = train_dataset.tensors
	weight_negatives = len(negatives)/torch.sum(negatives) - 1 #taux de négatif sur taux de positif 
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
	acc_val = []
	phi_sec_train = []
	phi_sec_val = []
	phi_train = []
	phi_val = []
	dh = display(printer("epoch : 0"), display_id=True)
	best_phi = 0
	best_state = None

	for i in range(nb_epochs):
		for sentences, emotions, negatives in tqdm(dataloader):
			optimizer.zero_grad()
			y, ys = rnn(sentences, return_both = True)
			loss = (1-secondary_proportion) * loss_function(y, emotions) + secondary_proportion * secondary_loss_function(ys, negatives) 
			loss.backward()
			optimizer.step()
		
		with torch.no_grad() :
			acc, phi, phi_sec = rnn.test(*train_dataset.tensors)
			acc_train.append(acc)
			phi_sec_train.append(phi_sec)
			phi_train.append(phi)
			
			acc, phi, phi_sec = rnn.test(*val_dataset.tensors)
			acc_val.append(acc)
			phi_sec_val.append(phi_sec)
			phi_val.append(phi)

		if phi_val[-1] > best_phi :
			best_phi = phi_val[-1]
			#need a deep-copy to revent values to keep being updated
			best_state = copy.deepcopy(rnn.state_dict()) 

		if dh :
			dh.update(printer(
				f""" 
					epoch : {i+1}
					Train : {acc_train[-1]}
					Test : {acc_val[-1]} 
				"""
			))   
		else : 
			# this is in case there is no interactive environment to avoid overflooding l'output
			print("epoch :",i+1) 

	#reload the best model that was obtained throughout the learning phase
	rnn.load_state_dict(best_state)

	# compute the confusion matrices
	cm_train = rnn.confusion_matrix(train_dataset)
	cm_val = rnn.confusion_matrix(val_dataset)
	cm_test = rnn.confusion_matrix(test_dataset)
	return (acc_train, acc_val), (phi_train, phi_val), (phi_sec_train, phi_sec_val), (cm_train, cm_val, cm_test)

def one_argument(func,arg):
	return func(*arg)


		
def test_rnn_on_multiple_cases_batch(cases, n= 10, num_workers = 1, **kwargs):

	acc = {}
	acc_sec = {}
	phi = {}
	cm = {}
	
	for name, case in cases.items() :
		print(name)
		case = {**kwargs, **case}
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
						[()] * n,
					)
				)
			)

		else : 
			acc[name] = []
			acc_sec[name] = []
			phi[name] = []
			cm[name] = []
						
			for _ in range(n):
				acc_one, acc_sec_one, phi_one, cm_one = train_rnn_one_argument_with_kwargs(())
				acc[name].append(acc_one)
				acc_sec[name].append(acc_sec_one)
				phi[name].append(phi_one)
				cm[name].append(cm_one)

	return  acc, acc_sec, phi, cm
#%%