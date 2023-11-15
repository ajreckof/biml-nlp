#%%
import torch
import utils
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, TensorDataset
from rnn import test_rnn_on_multiple_cases
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from inspect import cleandoc
from tqdm import tqdm
from IPython.display import clear_output
from matplotlib.colors import hsv_to_rgb

#%%

raw_phrases_train, raw_emotions_train = utils.load_file("dataset/train.txt")
raw_phrases_test, raw_emotions_test = utils.load_file("dataset/test.txt")
raw_phrases_val, raw_emotions_val = utils.load_file("dataset/val.txt")

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

chosen_min_freq = 5

print(f"we will be removing words of length strictly below {chosen_min_freq:.2f} which corresponds to {stats.percentileofscore(values,chosen_min_freq, kind= 'strict')}% of words ({stats.percentileofscore(values,chosen_min_freq, kind= 'strict')/100 *len(count_dict):.0f} words)")

vocab = build_vocab_from_iterator(raw_phrases_train, specials=["<unk>"], min_freq= chosen_min_freq)
vocab_no_min_freq = build_vocab_from_iterator(raw_phrases_train, specials=["<unk>"])

#%%
# Plot the distribution
plt.hist(values, bins=np.arange(min(values), max(values) + 1.5) - 0.5, edgecolor='black', alpha=0.7)
plt.yscale('log')
plt.xscale('log')
plt.show()
#%%
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

def select_n_max_elements(input_list, values, n):
	# Get indices of elements sorted by their values in descending order
	sorted_indices = sorted(range(len(input_list)), key=lambda i: values[i], reverse=True)

	# Take the first n indices and sort them in ascending order to maintain the original order
	selected_indices = sorted(sorted_indices[:n])

	# Extract the corresponding elements from the original list
	selected_elements = [input_list[i] for i in selected_indices]

	return selected_elements

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
			remove_unfrequent_word = [word for word in phrase if word in stoi]
			# label_encoded_phrase = [stoi[word] if word in stoi else s_unknown for word in phrase[:sentence_length]]
			phrase_tfidf = [tfidf_matrix[phrase_index,feature_names[word]] if word in feature_names else 0 for word in phrase]
			keep_most_interesting_words = select_n_max_elements(remove_unfrequent_word, phrase_tfidf, sentence_length)
			label_encoded_phrase = [stoi[word] for word in keep_most_interesting_words]
		else :
			label_encoded_phrase = [stoi[word] for word in phrase if word in stoi][:sentence_length]
		
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

train_dataset_no_tfidf = prepare(raw_phrases_train, sentence_length, raw_emotions_train, stoi, labelEncoder)
val_dataset_no_tfidf = prepare(raw_phrases_val, sentence_length, raw_emotions_val, stoi, labelEncoder)
test_dataset_no_tfidf = prepare(raw_phrases_test, sentence_length, raw_emotions_test, stoi, labelEncoder)

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
	n = 4,
	num_workers=4,
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
		
		fig = plt.figure(figsize = (21,14))
		for i, type in enumerate(types):
			ax = fig.add_subplot(2,3,1+i)
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
			ax = fig.add_subplot(2,3,4+i)
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




#%%
def avg(cm):
	return sum(cm)/len(cm)
def plot_cm(cm):
	base_val_or_train = ["train ", "val ", "test "]
	cm_info = { base_val_or_train[i] + case : avg(list(zip(*data_case))[i]) for case,data_case in cm.items() for i in range(3)}

	base_cases = cm.keys()
	for case in base_cases:
		fig = plt.figure(figsize = (20,5))
		fig.suptitle("matrice de confusion dans le cas " + case)
		plt.subplots_adjust(wspace=0.2)
		for i, sub_case in enumerate(base_val_or_train):
			ax = fig.add_subplot(1,3,i+1)
			ax.set_title(sub_case)
			ConfusionMatrixDisplay(cm_info[sub_case + case], display_labels = labelEncoder.classes_).plot(cmap = 'Blues', ax = ax, values_format='.3f')


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
			"batch_size" : 32,
		},
	},
	n = 10,	
	num_workers=5,
)
get_info_and_plot(acc,phi,phi_sec)
plot_cm(cm)


#%%
cases={
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
			"batch_size" : 32,
		}
}

for name,case in cases.items() :
	acc_case, phi_case, phi_sec_case, cm_case = test_rnn_on_multiple_cases(
		train_dataset = train_dataset, 
		val_dataset = val_dataset,
		test_dataset = test_dataset, 
		size_vocab =len(vocab),
		nb_epochs = 20,
		cases={name : case},
		n = 10,	
		num_workers=5,
	)
	clear_output(wait = True)
	acc.update(acc_case)
	phi.update(phi_case)
	phi_sec.update(phi_sec_case)
	cm.update(cm_case)
	get_info_and_plot(acc,phi,phi_sec)
	plot_cm(cm)







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