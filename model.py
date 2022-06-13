import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
import re
import unidecode
import string
import numpy as np
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
import cloudpickle
from sklearn import cluster, metrics
from sklearn import manifold, decomposition
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, LabelBinarizer
from sklearn import preprocessing,decomposition
from sklearn.model_selection import train_test_split,StratifiedKFold, \
                            ShuffleSplit, cross_val_score, KFold
from sklearn import model_selection,preprocessing,metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
import gensim
from gensim import corpora
from sklearn.decomposition import TruncatedSVD
import os
from corpus_norm import corpus_norm 

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')


script_dir = os.path.abspath(os.path.dirname( __file__ ))


df = pd.read_csv(script_dir + '/questions-tags.csv', index_col = 'Unnamed: 0')
print(df['Initial_Data'].head())
with open(script_dir + '/normalize.pickle', 'rb') as handle:
	normalizer = pickle.load(handle)
print(normalizer.normalize_corpus(df['Initial_Data']))
norm_features = normalizer.normalize_corpus(df['Initial_Data'])

with open(script_dir + '/vectorizer.pickle', 'rb') as handle:
	vect = pickle.load(handle)
v_features = vect.transform(norm_features)
print('V',v_features)
with open(script_dir + '/dimred.pickle', 'rb') as handle:
	svd = pickle.load(handle)
s_features = svd.transform(v_features)
print('S',s_features)

with open(script_dir + '/model.pickle', 'rb') as handle:
	model = pickle.load(handle)
prediction = model.predict(s_features)

with open(script_dir + '/binarizer.pickle', 'rb') as handle:
	mlb = pickle.load(handle)
print(mlb.inverse_transform(prediction))

#with open('normalize.pickle', 'wb') as handle:
#    pickle.dump(c, handle, protocol=pickle.HIGHEST_PROTOCOL)
#with open('dimred.pickle', 'wb') as handle:
#    pickle.dump(svd, handle, protocol=pickle.HIGHEST_PROTOCOL)
#with open('vectorizer.pickle', 'wb') as handle:
#    pickle.dump(cvect, handle, protocol=pickle.HIGHEST_PROTOCOL)
#with open('normalize.pickle', 'wb') as handle:
#    pickle.dump(c, handle, protocol=pickle.HIGHEST_PROTOCOL)
#with open('binarizer.pickle', 'wb') as handle:
#    pickle.dump(mlb, handle, protocol=pickle.HIGHEST_PROTOCOL)
