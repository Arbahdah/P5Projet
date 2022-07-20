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
from sklearn import cluster, metrics
from sklearn import manifold, decomposition
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
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
import joblib
import os

script_dir = os.path.abspath(os.path.dirname( __file__ ))

class corpus_norm:

    def __init__(self):
        #self.normalized_corpus = []
        self.stop_w = list(set(nltk.corpus.stopwords.words('english')))
        self.english_vocab = set(nltk.corpus.words.words()) 

    def clear_html_tags(self,html_doc) :
        if html_doc:
            soup = BeautifulSoup(html_doc, 'html.parser')
            return soup.get_text()

    def clean_path(self,value):
        value_clean = re.sub(r'http[:/\.a-z0-9]*','',value)
        return re.sub(r'/[/\.a-z0-9]*','',value_clean)

    def retrieve_code(self,value):
        result = []
        value = re.sub(r'\s', ' ', value)
        soup = BeautifulSoup(value, 'html.parser')
        if soup.find_all('code'):
            for elt in soup.find_all('code'):
                result.append(elt.get_text())
        return result

    def suppress_code(self,value):
        init_value = value
        value = re.sub(r'\s', ' ', value)
        soup = BeautifulSoup(value, 'html.parser')
        if soup.find_all('code'):
            for elt in soup.find_all('code'):
                value = str(value).replace(str(elt),' ') 
            if value == init_value:
                isGoodString = True
                v_clean = ''
                for v in value.split(' '):
                    if '<code>' in v:
                        isGoodString = False
                    if isGoodString:
                        v_clean += v
                    if '</code>' in v:
                        isGoodString = True
                value = v_clean
        return value

    def clean_web_addresses(self,value):
        init_value = value
        value = re.sub(r'\s', ' ', value)
        soup = BeautifulSoup(value, 'html.parser')
        if soup.find_all('a'):
            for elt in soup.find_all('a'):
                value = value.replace(str(elt),' ')
        return value

    def clear_tags(self,value):
        value_0 = self.clean_web_addresses(value)
        value_1 = self.suppress_code(value_0)
        value_2 = self.clear_html_tags(value_1)
        if value_2:
            return value_2.replace('\n','')

    def tokenizer_fct(self,sentence) :
        word_tokens = word_tokenize(sentence)
        return word_tokens

    def stop_word_filter_fct(self,list_words, list_tags,list_sw = []) :
        stop_w_i = self.stop_w + list(set(list_sw))
        filtered_w = [w.replace(' ','')  \
                  for w in list_words \
                  if w.replace(' ','') not in stop_w_i or w in list_tags]
        return filtered_w

    def casefold(self,list_values):
        list_result = []
        for value in list_values:
            list_result.append(value.casefold())
        return sorted(set(list_result))

    def replace_chars(self,list_values):
        list_result = []
        chars = re.escape(string.punctuation)
        for value in list_values:
            #if value not in list_tags:
            status = bool(re.search(r"[a-z]{3,}", value))
            if status:
                value_clean = re.sub(r'^['+chars+']+','',value)
                list_result.append(value_clean)
        return list_result

    def words_more_2_chars(self,list_values,tags_list):
        list_result = []
        for value in list_values:
            if len(value) > 2 or value in tags_list:
                list_result.append(value)
        return list_result

    def words_more_3_chars(self,list_values,tags_list=None):
        list_result = []
        for value in list_values:
            if len(value) > 3:
             # or value in tags_list:
                list_result.append(value)
        return list_result

    # Lemmatizer (base d'un mot)
    def lemma_fct(self,list_words):
        lemmatizer = WordNetLemmatizer()
        lem_w = []
        nltk_tags = nltk.pos_tag(list_words)
        for (w,nltk_tag) in nltk_tags:
            w = w.replace(' ','')
            l_r = w
            if len(lemmatizer.lemmatize(w,wordnet.ADJ)) < len(w):
                l_r = lemmatizer.lemmatize(w,wordnet.ADJ)
            elif len(lemmatizer.lemmatize(w,wordnet.VERB)) < len(l_r):
                l_r = lemmatizer.lemmatize(w,wordnet.VERB)
            elif len(lemmatizer.lemmatize(w,wordnet.NOUN)) < len(l_r):
                l_r = lemmatizer.lemmatize(w,wordnet.NOUN)
            elif len(lemmatizer.lemmatize(w,wordnet.ADV)) < len(l_r):
                l_r = lemmatizer.lemmatize(w,wordnet.ADV)    
            elif nltk_tag.startswith('J'):
                l_r = lemmatizer.lemmatize(w,wordnet.ADJ)
            elif nltk_tag.startswith('V'):
                l_r = lemmatizer.lemmatize(w,wordnet.VERB)
            elif nltk_tag.startswith('N'):
                l_r = lemmatizer.lemmatize(w,wordnet.NOUN)
            elif nltk_tag.startswith('R'):
                l_r = lemmatizer.lemmatize(w,wordnet.ADV)
        #else:
        #    t_wordnet = wordnet.NOUN
        #word = lemmatizer.lemmatize(w,t_wordnet)
            if len(l_r) > 2:
                lem_w.append(l_r)
        return lem_w

    #Vérifier les caractères
    def check_chars(self,list_values,tags_list=None):
        list_result = []
        for value in list_values:
            value_clean = unidecode.unidecode(value)
            status = bool(re.match(r"^[a-z]+[a-z0-9]+$", value_clean))
            if status:
            # and value_clean not in tags_list:
                list_result.append(value_clean)
        #elif value_clean in tags_list:
        #    list_result.append(value)
        return list_result

    def clean_punctuation(self,list_values,tags_list=None):
        list_result = []
        chars = re.escape(string.punctuation)
        for value in list_values:
        #if value not in tags_list:
            value_clean = re.sub(r'['+chars+']+',' ',value)
        #else:
        #    value_clean = value
            value_c = value_clean.split(' ')
            for v in value_c:
                if v:
                    list_result.append(v)
        return list_result
    #return value_clean

    def list_to_string(self,list_in):
        return ' '.join(list_in)

    def filter_tags(self,text,list_c):
        result = text
        for value in text:
            if value not in list_c:
                result = []
        return result

    def filter_words(self,list_w, list_t, tags_list):
        list_r = []
        for value in list_w:
            if value in list_t or value in tags_list:
                list_r.append(value)
        return list_r

    def sort_words(self,list_w):
        l_r = []
        for elt in list_w:
            if elt not in l_r:
                l_r.append(elt)
        return l_r

    def custom_import_stopwords(self,filename):
        infile = open(filename,encoding = 'utf8') 
        in_stopword_list = infile.readlines() 
        infile.close()
        stopword_list = []
        for elt in in_stopword_list:
            stopword_list.append(re.sub('\n','',elt))
        return stopword_list

    def normalize_corpus(self,corpus):
        normalized_corpus = []
        list_sw = self.custom_import_stopwords(script_dir + '/resources/stop_words_english.txt')
        for doc in corpus:
            doc = self.suppress_code(doc)
            doc = self.clean_web_addresses(doc)
            doc = self.clear_html_tags(doc)
            doc = self.tokenizer_fct(doc)
            doc = self.casefold(doc)
            doc = self.clean_punctuation(doc)
            doc = self.check_chars(doc)
            doc = self.stop_word_filter_fct(doc,list_sw)
            doc = self.words_more_3_chars(doc)
            doc = self.lemma_fct(doc)
            doc = self.sort_words(doc)
            doc = self.list_to_string(doc)
            normalized_corpus.append(doc)
        return normalized_corpus
