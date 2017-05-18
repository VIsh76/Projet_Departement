import pandas as pd
import numpy as np
import string
import cPickle as pickle
import os, sys
from bidict import bidict
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem.snowball import FrenchStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans

data_path = "../Data_M[2005-2017].csv"
label_path = "../Dico_M[2005-2017].csv"

def get_tokens(text, toktok):
    lowers = text.lower()
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    no_punctuation = lowers.translate(remove_punctuation_map)
    tokens = toktok.tokenize(no_punctuation)
    return tokens

fp_label = pd.read_csv(label_path, encoding = 'utf-8')
var_name = fp_label['def']
id_bank = np.array(fp_label['IDBANK'], dtype = np.str)
dict_id_name = bidict(zip(np.arange(id_bank.shape[0]),id_bank))

toktok = ToktokTokenizer()
stemmer = FrenchStemmer()

var_tokens = []

for i in range(var_name.size):
    tokens = get_tokens(var_name[i], toktok)
    new_tokens = []
    
    for j in range(len(tokens)):
        if not tokens[j].isnumeric():
            new_tokens.append(stemmer.stem(tokens[j]))
    var_tokens.append(' '.join(new_tokens))

print len(var_tokens)

tf_count = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000)
tf = tf_count.fit_transform(var_tokens)

kmeans = KMeans(n_clusters=10, random_state=0).fit(tf)
_, count = np.unique(kmeans.labels_, return_counts=True)

print count

y_name = np.array(id_bank, dtype = np.int)
tf_class = dict(zip(y_name, kmeans.labels_))

with open('../Data/kmeans.pkl', 'wb') as outfile:
    pickle.dump(tf_class, outfile)
