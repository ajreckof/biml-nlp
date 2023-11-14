#%%
import torch
import utils
from torchtext.vocab import build_vocab_from_iterator
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from rnn import test_rnn_with_or_without_emotions_weights
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from IPython.display import clear_output, display
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from inspect import cleandoc

from tqdm import tqdm
import time
#%%

raw_phrases_train, raw_emotions_train = utils.load_file("dataset/train.txt")
raw_phrases_test, raw_emotions_test = utils.load_file("dataset/test.txt")
phrases_val, emotions_val = utils.load_file("dataset/val.txt")

#%%
class printer(str):
    def __repr__(self):
       return cleandoc(self)
    def __print__(self):
       return cleandoc(self)

#%%
# Create an empty dictionary to store counts
count_dict = {}

# Iterate through the list and update the counts in the dictionary
for phrase in raw_phrases_train:
	for word in phrase :
		if word in count_dict:
			count_dict[word] += 1
		else:
			count_dict[word] = 1
values = list(count_dict.values())
for i in range(1,19):
	print(i,":" ,stats.percentileofscore(values, i, kind= 'strict'))
# Plot the distribution
plt.hist(values, bins=np.arange(min(values), max(values) + 1.5) - 0.5, edgecolor='black', alpha=0.7)
plt.yscale('log')
plt.xscale('log')
plt.show()
chosen_min_freq = 2

print(f"we will be removing words of length strictly below {chosen_min_freq} which corresponds to {stats.percentileofscore(values,chosen_min_freq, kind= 'strict')}% of words ({stats.percentileofscore(values,chosen_min_freq, kind= 'strict') *len(count_dict)})")

# %%
vocab = build_vocab_from_iterator(raw_phrases_train, specials=["<unk>"], min_freq= chosen_min_freq)

#%%
len_phrase = [len(phrase) for phrase in raw_phrases_train]
stats.describe(len_phrase)
plt.hist(len_phrase, bins=np.arange(min(len_phrase), max(len_phrase) + 1.5) - 0.5, edgecolor='black', alpha=0.7)
plt.show()
stoi = vocab.get_stoi()
len_phrase_without_unfrequent = [len([word for word in phrase if word in stoi]) for phrase in raw_phrases_train]
plt.hist(len_phrase_without_unfrequent, bins=np.arange(min(len_phrase_without_unfrequent), max(len_phrase_without_unfrequent) + 1.5) - 0.5, edgecolor='black', alpha=0.7)
plt.show()
# longueur moyenne des phrases = 20
# variance = 120


#%%

negatives_indicators = [
	" not ",
	"n't ",
	" never ",
	" neither ",
	" no ",
	" none ",
	" nobody ",
	" nowhere ",
	" nothing ",
]
def get_negative_from_phrase(phrase):
	full_phrase = " " + " ".join(phrase).lower() + " "
	return any(indicator in full_phrase for indicator in negatives_indicators)

def select_n_max_elements(input_list, values, n):
	# Get indices of elements sorted by their values in descending order
	sorted_indices = sorted(range(len(input_list)), key=lambda i: values[i], reverse=True)

	# Take the first n indices and sort them in ascending order to maintain the original order
	selected_indices = sorted(sorted_indices[:n])

	# Extract the corresponding elements from the original list
	selected_elements = [input_list[i] for i in selected_indices]

	return selected_elements

def prepare(phrases, sentence_length, emotions, stoi, labelEncoder, tf_idf):
	label_encoded_phrases = []
	negatives = []
	s_unknown = stoi["<unk>"]
	joined_phrases = [" ".join(phrase) for phrase in phrases]
	tfidf_matrix = tf_idf.transform(joined_phrases)
	feature_names = tf_idf.get_feature_names_out()
	for phrase_index, phrase in tqdm(enumerate(phrases), total=len(phrases)) :
		remove_unfrequent_word = [word for word in phrase if word in stoi]
		# label_encoded_phrase = [stoi[word] if word in stoi else s_unknown for word in phrase[:sentence_length]]
		phrase_tfidf = [tfidf_matrix[phrase_index,np.where(feature_names == word)] if word in feature_names else 0 for word in phrase]
		keep_most_interesting_words = select_n_max_elements(remove_unfrequent_word, phrase_tfidf, sentence_length)
		label_encoded_phrase = [stoi[word] for word in keep_most_interesting_words]
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
labelEncoder = LabelEncoder()
labelEncoder.fit(raw_emotions_train)

tf_idf = TfidfVectorizer()
joined_phrases = [" ".join(phrase) for phrase in raw_phrases_train]
tf_idf.fit(joined_phrases)

# Dataset d'entrainement
train_dataset = prepare(raw_phrases_train, sentence_length, raw_emotions_train, stoi, labelEncoder, tf_idf)
test_dataset = prepare(raw_phrases_test, sentence_length, raw_emotions_test, stoi, labelEncoder, tf_idf)



# %%
acc, acc_sec, phi, cm = test_rnn_with_or_without_emotions_weights(
	train_dataset, 
	test_dataset, 
	len(vocab),
	cases={
		"default": {},
	},
	n = 4,
	num_workers=4,
)

# %%
acc, acc_sec, phi, cm = test_rnn_with_or_without_emotions_weights(
	train_dataset, 
	test_dataset, 
	len(vocab),
	cases={
		"default": {},
		"without weight": {"with_emotions_weight": False},
		"without secondary": {"secondary_proportion" : 0},
		"high embeddings": {"embed_size" : 300, "hidden_size" : 300, "batch_size": 24},
		"low embeddings": {"embed_size" : 50, "hidden_size" : 50, "batch_size": 4},
	},
	n = 4,	
	num_workers=4,
)

#%%
base_cases = acc.keys()
base_test_or_train = ["test ", "train "]
cases = [test_or_train+case for case in base_cases for test_or_train in base_test_or_train]
val = np.linspace(0, 1, len(cases))
cmap = plt.get_cmap('rainbow')  
color_list = [cmap(value) for value in val]
colors = {case : color for case,color in zip(cases,color_list)}
percentiles_base = [0.1, 1, 5, 10, 25, 33, 40,45]
percentiles_values = percentiles_base + [ 100 - x for x in reversed(percentiles_base)]

def get_info_and_plot(data, cases = cases):
	info = { base_test_or_train[i] + case : list(zip(*list(zip(*data_case))[i])) for case,data_case in data.items() for i in range(2)}
	
	mean = {}
	median = {}
	percentiles = {}
	
	for case in cases :
		percentiles[case] = list(zip(*[np.percentile(x, percentiles_values) for x in info[case]]))
		median[case] = [np.median(x) for x in info[case]]
		mean[case] = [np.mean(x) for x in info[case]]

	fig = plt.figure(figsize = (10,5))

	ax = fig.add_subplot(1,2,1)
	ax.grid(visible= True, which='both')
	ax.set_xlabel("epoch")
	ax.set_ylabel("médiane")
	ax.set_ylim(0,1)
	for case in cases :
			ax.plot(np.arange(len(median[case])), median[case], label = case, color = colors[case])
			for i in range(len(percentiles_base)) : 
				ax.fill_between(np.arange(len(median[case])), percentiles[case][i], percentiles[case][-i-1], color = colors[case], alpha = 0.1)
	ax.legend(loc='upper left')

	ax = fig.add_subplot(1,2,2)
	ax.grid(visible= True, which='both')
	ax.set_xlabel("epoch")
	ax.set_ylabel("moyenne")
	ax.set_ylim(0,1)
	for case in cases :
		ax.plot(np.arange(len(mean[case])), mean[case], label = case, color = colors[case])
	ax.legend(loc='upper left')

get_info_and_plot(acc)
get_info_and_plot(acc_sec)
get_info_and_plot(phi)



#%%
_,emotions, _ = train_dataset.tensors
emotions_count = torch.bincount(emotions).to(torch.double)
weight_emotions = len(emotions)/len(emotions_count)/emotions_count

print(emotions_count)
print(weight_emotions)

_,emotions, negatives = test_dataset.tensors
emotions_count = torch.bincount(emotions).to(torch.double)
weight_emotions = len(emotions)/len(emotions_count)/emotions_count

print(emotions_count)
print(weight_emotions)

#%%
cm_with_weight_train

#%%
mean_cm_with_weight_train = sum(cm_with_weight_train)/len(cm_with_weight_train)
mean_cm_with_weight_test = sum(cm_with_weight_test)/len(cm_with_weight_test)
mean_cm_without_weight_train = sum(cm_without_weight_train)/len(cm_without_weight_train)
mean_cm_without_weight_test = sum(cm_without_weight_test)/len(cm_without_weight_test)


#%%
mean_cm_with_weight_train = np.transpose(mean_cm_with_weight_train)
mean_cm_with_weight_test = np.transpose(mean_cm_with_weight_test)
mean_cm_without_weight_train = np.transpose(mean_cm_without_weight_train)
mean_cm_without_weight_test = np.transpose(mean_cm_without_weight_test)

#%%

def normalize(cm):
	row_sums = np.sum(cm, axis=1)
	return cm/row_sums

normalize_mean_cm_with_weight_train = normalize(mean_cm_with_weight_train)
normalize_mean_cm_with_weight_test = normalize(mean_cm_with_weight_test)
normalize_mean_cm_without_weight_train = normalize(mean_cm_without_weight_train)
normalize_mean_cm_without_weight_test = normalize(mean_cm_without_weight_test)
#%%

ConfusionMatrixDisplay(normalize_mean_cm_with_weight_train, display_labels = labelEncoder.classes_).plot(cmap = 'Blues')
ConfusionMatrixDisplay(normalize_mean_cm_with_weight_test, display_labels = labelEncoder.classes_).plot(cmap = 'Blues')
ConfusionMatrixDisplay(normalize_mean_cm_without_weight_train, display_labels = labelEncoder.classes_).plot(cmap = 'Blues')
ConfusionMatrixDisplay(normalize_mean_cm_without_weight_test, display_labels = labelEncoder.classes_).plot(cmap = 'Blues')
#%%
print(printer(
	f"""
		weight train : { sum(acc_with_weight_train) / n }({ np.std(acc_with_weight) })
		weight test: { sum(acc_with_weight_test) / n }({ np.std(acc_with_weight) })
		no weight train: { sum(acc_without_weight_train) / n }({ np.std(acc_without_weight) })
		no weight test: { sum(acc_without_weight_test) / n }({ np.std(acc_without_weight) })
	"""
))




# %%
acc_test2, acc_train2, acc_secondary_test2, acc_secondary_train2, phi_test2, phi_train2, cm_train2, cm_test2 = test_rnn_with_or_without_emotions_weights(train_dataset, test_dataset, len(vocab), n = 4)







#%% for safekeeping as it is not good at all 

# rnn = RNN(input_size=len(vocab), embed_size=20, hidden_size=20, output_size=len(set(raw_emotions_train)), batch_size=batch_size)
# loss_function = torch.nn.CrossEntropyLoss()
# secondary_loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight_negatives]))
# optimizer = torch.optim.Adam(rnn.parameters(),lr=lr)
# acc_train = []
# acc_test = []
# acc_train_secondary = []
# acc_test_secondary = []
# print("epoch : 0")
# for i in range(nb_epochs):
#     for sentences, emotions, _ in dataloader:
#         optimizer.zero_grad()
#         y = rnn(sentences)
#         loss = loss_function(y, emotions)
#         loss.backward()
#         optimizer.step()
	
#     with torch.no_grad() :
#         acc_test.append(rnn.test(*test_dataset.tensors))
#         acc_train.append(rnn.test(*train_dataset.tensors))
#         clear_output(wait= True)
	
#     for sentences, _, negatives in dataloader:
#         optimizer.zero_grad()
#         y = rnn(sentences, is_secondary = True)
#         loss = secondary_loss_function(y, negatives) 
#         loss.backward()
#         optimizer.step()
	
#     with torch.no_grad() :
#         acc_test_secondary.append(rnn.test(*test_dataset.tensors))
#         acc_train_secondary.append(rnn.test(*train_dataset.tensors))
#         clear_output(wait= True)
	

#     print(
#         f""" 
#         epoch : {i+1}
#         Train : {acc_train[-1]}
#         Test : {acc_test[-1]} 
#         Train secondary : {acc_train_secondary[-1]}
#         Test secondary: {acc_test_secondary[-1]} 
#         """
#     )
# #%%
# plot(acc_train, acc_test)
# plot(acc_train_secondary, acc_test_secondary)

# # %%