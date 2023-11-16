#%%
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, TensorDataset

from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, matthews_corrcoef,ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer


import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

from multiprocessing import Pool
from tqdm import tqdm
import signal
from functools import partial


from IPython.display import display
import copy

from inspect import cleandoc
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
		
		
		self.i2e = nn.Linear(input_size, embed_size, dtype=torch.double) # embedding
		self.i2s = nn.Linear(embed_size + hidden_size, 1, dtype=torch.double) # secondary output
		self.i2o = nn.Linear(embed_size + hidden_size, output_size, dtype=torch.double) # primary output
		self.i2h = nn.Linear(embed_size + hidden_size, hidden_size, dtype=torch.double) # hidden

		nn.init.xavier_uniform_(self.i2e.weight)
		nn.init.xavier_uniform_(self.i2s.weight)
		nn.init.xavier_uniform_(self.i2o.weight)
		nn.init.xavier_uniform_(self.i2h.weight)


	def forward(self,input, is_secondary = False, return_both = False):
		# passe successivement dans le modèle chaque mot des phrases présentes dans input

		# récupération de paramètre via la shape de l'entré
		batch_size, num_columns = input.shape
		
		hidden = self.initHidden(batch_size)
		# Itérer sur les mots en s’arrêtant un mot avant la fin
		for i in range(num_columns - 1):
			batch_word = input[:, i]
			# ici le forward word renvoie que le hidden car on a pas besoin de l'output à chaque mot mais que au dernier mot
			hidden = self.forward_word(batch_word, hidden)

		# on fait passer le dernier mot et cette fois on sort les output au lieu de hidden
		batch_word = input[:, num_columns-1]
		return self.forward_word(batch_word, hidden, is_final = True, return_both = return_both, is_secondary= is_secondary)

	def forward_word(self, input, hidden, is_final = False, is_secondary = False, return_both = False):
		# passe un mot dans le modèle

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
		#initialise avec des zeros la valeur de la couche caché
		return torch.zeros(batch_size, self.hidden_size, dtype=torch.double)
	
	def test(self, phrases, emotions, negatives):
		emotions_output, negatives_output = self(phrases, return_both = True)
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
	# calcul des différents poids 
	_,emotions, negatives = train_dataset.tensors
	weight_negatives = len(negatives)/torch.sum(negatives) - 1 #taux de négatif sur taux de positif 
	emotions_count = torch.bincount(emotions).to(torch.double)
	weight_emotions = len(emotions)/len(emotions_count)/emotions_count

	# création des loss et optimiseur
	if with_emotions_weight :
		loss_function = torch.nn.CrossEntropyLoss(weight=weight_emotions)
	else : 
		loss_function = torch.nn.CrossEntropyLoss()
	secondary_loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight_negatives]))
	optimizer = torch.optim.AdamW(rnn.parameters(),lr=lr)
	dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

	rnn = RNN(input_size=size_vocab, embed_size=embed_size, hidden_size=hidden_size, output_size=len(weight_emotions), batch_size=batch_size)

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
			#need a deep-copy to prevent values to keep being updated
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
			# this is in case there is no interactive environment to avoid overflowing the output
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


		
def test_rnn_on_multiple_cases(cases, n= 10, num_workers = 1, **kwargs):
	"""
		Dans l'état du fichier (merge de plusieurs fichier) ne pas lancer cette fonction avec num_workers différent de 1
	"""
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
			# Dans ce cas on ne fait pas de multiprocessing
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
# Lecture des différents fichiers et mise sous forme de dataset
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


raw_phrases_train, raw_emotions_train = load_file("dataset/train.txt")
raw_phrases_test, raw_emotions_test = load_file("dataset/test.txt")
raw_phrases_val, raw_emotions_val = load_file("dataset/val.txt")

#%%
# Étude de la distribution du nombre d'occurrence de chaque mot
count_dict = {}

for phrase in raw_phrases_train:
	for word in phrase :
		if word in count_dict:
			count_dict[word] += 1
		else:
			count_dict[word] = 1
values = list(count_dict.values())
for i in range(1,19):
	print(i,":" ,stats.percentileofscore(values, i, kind= 'strict'))

chosen_min_freq = 5

print(f"we will be removing words of length strictly below {chosen_min_freq:.2f} which corresponds to {stats.percentileofscore(values,chosen_min_freq, kind= 'strict')}% of words ({stats.percentileofscore(values,chosen_min_freq, kind= 'strict')/100 *len(count_dict):.0f} words)")

vocab = build_vocab_from_iterator(raw_phrases_train, specials=["<unk>"], min_freq= chosen_min_freq)
vocab_no_min_freq = build_vocab_from_iterator(raw_phrases_train, specials=["<unk>"])

#%%
# histogramme de la distribution en log-log scale pour pouvoir voir la loi de puissance

plt.hist(values, bins=np.arange(min(values), max(values) + 1.5) - 0.5, edgecolor='black', alpha=0.7)
plt.yscale('log')
plt.xscale('log')
plt.show()
#%%
# Comparaison de al taille des phrases avant et après avori enlevé les mots les moins fréquents
len_phrase = [len(phrase) for phrase in raw_phrases_train]
print(stats.describe(len_phrase))
plt.hist(len_phrase, bins=np.arange(min(len_phrase), max(len_phrase) + 1.5) - 0.5, edgecolor='black', alpha=0.7)
plt.show()
stoi = vocab.get_stoi()
len_phrase_without_unfrequent = [len([word for word in phrase if word in stoi]) for phrase in raw_phrases_train]
plt.hist(len_phrase_without_unfrequent, bins=np.arange(min(len_phrase_without_unfrequent), max(len_phrase_without_unfrequent) + 1.5) - 0.5, edgecolor='black', alpha=0.7)
plt.show()
print(stats.describe(len_phrase_without_unfrequent))
# longueur moyenne des phrases = 20
# variance = 120


#%%
# Labelisation des phrases en négative ou non 

negatives_indicators = [
	"not",
	"t",
	"dont",
	"didnt",
	"cant"
	"never",
	"neither",
	"no",
	"none",
	"nobody",
	"nowhere",
	"nothing",
]
def get_negative_from_phrase(phrase):
	return any(indicator in phrase for indicator in negatives_indicators)

# fonction prenant les n éléments de la list input_list ayant le plus grand score dans la liste values en conservant l'ordre
def select_n_max_elements(input_list, values, n):
	# Get indices of elements sorted by their values in descending order
	sorted_indices = sorted(range(len(input_list)), key=lambda i: values[i], reverse=True)

	# Take the first n indices and sort them in ascending order to maintain the original order
	selected_indices = sorted(sorted_indices[:n])

	# Extract the corresponding elements from the original list
	selected_elements = [input_list[i] for i in selected_indices]

	return selected_elements

#préparation des jeu de données pour utilisation par le rnn
# les encoder/indicateur sont fit et donnée déjà entraîné à cette fonction
def prepare(phrases, sentence_length, emotions, stoi, labelEncoder, tf_idf = None):
	label_encoded_phrases = []
	negatives = []
	s_unknown = stoi["<unk>"]
	joined_phrases = [" ".join(phrase) for phrase in phrases]

	if tf_idf :
		tfidf_matrix = tf_idf.transform(joined_phrases)
		feature_names = {word: index for index, word in enumerate(tf_idf.get_feature_names_out())}
	
	for phrase_index, phrase in tqdm(enumerate(phrases), total=len(phrases)) :
		if tf_idf :
			remove_unfrequent_word = [word for word in phrase if word in stoi] # retiré les mots non présent dans le vocabulaire
			phrase_tfidf = [tfidf_matrix[phrase_index,feature_names[word]] if word in feature_names else 0 for word in phrase] # obtenir leur valeur d'après TF-IDF
			keep_most_interesting_words = select_n_max_elements(remove_unfrequent_word, phrase_tfidf, sentence_length) # garder les n avec le plus grand TF-IDF en gardant l'ordre de la phrase
			label_encoded_phrase = [stoi[word] for word in keep_most_interesting_words] # encode with the vocab 
		else :
			label_encoded_phrase = [stoi[word] for word in phrase if word in stoi][:sentence_length] #without tf-idf only encoding and keeping the n first that are in the vocab
		
		label_encoded_phrases.append( label_encoded_phrase + [s_unknown] * (sentence_length - len(label_encoded_phrase)))
		negatives.append(get_negative_from_phrase(phrase))
	# Préparation des données des émotions
	label_encoded_emotions = labelEncoder.transform(emotions)
	
	# Dataset
	return TensorDataset(torch.tensor(label_encoded_phrases), torch.tensor(label_encoded_emotions), torch.tensor(negatives, dtype= torch.double).unsqueeze(1))

#%%
# Préparation des données
sentence_length = 15 

stoi = vocab.get_stoi()
stoi_no_min_freq = vocab_no_min_freq.get_stoi()

labelEncoder = LabelEncoder()
labelEncoder.fit(raw_emotions_train)

tf_idf = TfidfVectorizer()
joined_phrases = [" ".join(phrase) for phrase in raw_phrases_train]
tf_idf.fit(joined_phrases)

# Dataset d'entrainement
train_dataset = prepare(raw_phrases_train, sentence_length, raw_emotions_train, stoi, labelEncoder, tf_idf)
val_dataset = prepare(raw_phrases_val, sentence_length, raw_emotions_val, stoi, labelEncoder, tf_idf)
test_dataset = prepare(raw_phrases_test, sentence_length, raw_emotions_test, stoi, labelEncoder, tf_idf)

# Dataset pour comparaison
train_dataset_no_tfidf = prepare(raw_phrases_train, sentence_length, raw_emotions_train, stoi, labelEncoder)
val_dataset_no_tfidf = prepare(raw_phrases_val, sentence_length, raw_emotions_val, stoi, labelEncoder)
test_dataset_no_tfidf = prepare(raw_phrases_test, sentence_length, raw_emotions_test, stoi, labelEncoder)

# Dataset pour comparaison (non-inclus rapport par manque de place)
train_dataset_no_min_freq = prepare(raw_phrases_train, sentence_length, raw_emotions_train, stoi_no_min_freq, labelEncoder, tf_idf)
val_dataset_no_min_freq = prepare(raw_phrases_val, sentence_length, raw_emotions_val, stoi_no_min_freq, labelEncoder, tf_idf)
test_dataset_no_min_freq = prepare(raw_phrases_test, sentence_length, raw_emotions_test, stoi_no_min_freq, labelEncoder, tf_idf)



# %%
acc, phi, phi_sec, cm = test_rnn_on_multiple_cases(
	train_dataset = train_dataset, 
	val_dataset = val_dataset,
	test_dataset = test_dataset, 
	size_vocab =len(vocab),
	cases={"default": {}},
	n = 10,
	# num_workers=4,
)


#%%

percentiles_base = [0.1, 1, 5, 10, 25, 33, 40,45]
percentiles_values = percentiles_base + [ 100 - x for x in reversed(percentiles_base)]

def get_info_and_plot(*data_by_type, default_case = "defaut"):
	base_cases = data_by_type[0].keys()
	val_or_trains = ["train ", "val "]
	types = ["précision ", "coefficient phi ", "coefficient phi sur tache secondaire"]

	info = { 
		types[type] + val_or_trains[val_or_train] + case : list(zip(*data)) 
		for type, data_by_case in enumerate(data_by_type)
		for case in base_cases 
		for val_or_train,data in enumerate(list(zip(*data_by_case[case])))
	}
	
	mean = {}
	median = {}
	percentiles = {}
	

	for case in info :
		percentiles[case] = list(zip(*[np.percentile(x, percentiles_values) for x in info[case]]))
		median[case] = [np.median(x) for x in info[case]]
		mean[case] = [np.mean(x) for x in info[case]]

	# Create evenly spaced hues for both groups
	hues = np.linspace(0, 1, len(base_cases), endpoint=False)

	# Set the same saturation and lightness for both groups

	saturation_val = 0.8
	lightness_val = 0.8
	saturation_train = 0.5
	lightness_train = 0.5

	# Create colors for both groups
	colors = {
		**{"val " + base_case	: hsv_to_rgb([hue, saturation_val, lightness_val]) for base_case, hue in zip(base_cases,hues)},
		**{"train " + base_case	: hsv_to_rgb([hue, saturation_train, lightness_train]) for base_case, hue in zip(base_cases,hues)}
	}

	for base_case in base_cases :
		
		
		for i, type in enumerate(types):
			fig = plt.figure(figsize = (5,5))
			ax = fig.add_subplot(1,1,1)
			ax.grid(visible= True, which='both')
			ax.set_xlabel("epoch")
			ax.set_ylabel("médiane")
			ax.set_ylim(0,1)
			for val_or_train in val_or_trains:
				
				case = type + val_or_train + base_case
				case_no_type = val_or_train + base_case
				ax.plot(np.arange(len(median[case])), median[case], label = case_no_type, color = colors[case_no_type])
				for j in range(len(percentiles_base)) : 
					ax.fill_between(np.arange(len(median[case])), percentiles[case][j], percentiles[case][-j-1], color = colors[case_no_type], alpha = 0.1)
				

				if base_case != default_case:
					case = type + val_or_train + default_case
					case_no_type = val_or_train + default_case
					ax.plot(np.arange(len(median[case])), median[case], label = case_no_type, color = colors[case_no_type])
					for j in range(len(percentiles_base)) : 
						ax.fill_between(np.arange(len(median[case])), percentiles[case][j], percentiles[case][-j-1], color = colors[case_no_type], alpha = 0.1)
				
			ax.legend(loc='lower right')

			fig.savefig("output/" + type + base_case + " median.png", bbox_inches= 'tight')
			plt.close(fig)

			fig = plt.figure(figsize = (5,5))
			ax = fig.add_subplot(1,1,1)
			ax.grid(visible= True, which='both')
			ax.set_xlabel("epoch")
			ax.set_ylabel("moyenne")
			ax.set_ylim(0,1)

			for val_or_train in val_or_trains:

				case = type + val_or_train + base_case
				case_no_type = val_or_train + base_case
				ax.plot(np.arange(len(mean[case])), mean[case], label = case_no_type, color = colors[case_no_type])
			
				if base_case != default_case:
					case = type + val_or_train + default_case
					case_no_type = val_or_train + default_case
					ax.plot(np.arange(len(mean[case])), mean[case], label = case_no_type, color = colors[case_no_type])
			
			ax.legend(loc='lower right')
			fig.savefig("output/" + type + base_case + " mean.png", bbox_inches= 'tight')
			plt.close(fig)




#%%
def avg(cm):
	return sum(cm)/len(cm)
def plot_cm(cm):
	base_val_or_train = ["jeu d entraînement ", "jeu de validation ", "jeu de test "]
	cm_info = { base_val_or_train[i] + case : avg(list(zip(*data_case))[i]) for case,data_case in cm.items() for i in range(3)}

	base_cases = cm.keys()
	for case in base_cases:
		for i, sub_case in enumerate(base_val_or_train):
			fig = plt.figure(figsize = (6,5))
			ax = fig.add_subplot(1,1,1)
			ConfusionMatrixDisplay(cm_info[sub_case + case], display_labels = labelEncoder.classes_).plot(cmap = 'Blues', ax = ax, values_format='.3f')
			fig.savefig("output/" + sub_case + case + " cm.png", bbox_inches= 'tight')
			plt.close(fig)
		plt.close(fig)



# %%
acc, phi, phi_sec, cm = test_rnn_on_multiple_cases(
	train_dataset = train_dataset, 
	val_dataset = val_dataset,
	test_dataset = test_dataset, 
	size_vocab =len(vocab),
	nb_epochs = 20,
	cases={
		"defaut"				: {},
		"sans poids"			: {"with_emotions_weight": False},
		"sans tâche secondaire"	: {"secondary_proportion" : 0},
		"high embeddings"		: {"embed_size" : 200, "hidden_size" : 200, "batch_size": 16},
		"low embeddings"		: {"embed_size" : 50, "hidden_size" : 50, "batch_size": 4},
		"sans tf-idf"			: {
			"train_dataset" : train_dataset_no_tfidf, 
			"val_dataset" : val_dataset_no_tfidf, 
			"test_dataset" : test_dataset_no_tfidf,
		},
		"sans min freq"			: {
			"train_dataset" : train_dataset_no_min_freq, 
			"val_dataset" : val_dataset_no_min_freq, 
			"test_dataset" : test_dataset_no_min_freq, 
			"size_vocab" : len(vocab_no_min_freq),
			"batch_size" : 64,
		},
	},
	n = 10,	
	# num_workers=5,
)
get_info_and_plot(acc,phi,phi_sec)
plot_cm(cm)
