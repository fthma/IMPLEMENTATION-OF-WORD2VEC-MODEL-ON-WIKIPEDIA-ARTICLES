# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 04:27:10 2020

@author: mir_h
"""

import re  # For preprocessing
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency

import spacy  # For preprocessing

import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
file=open("shortened_wikifile.txt",errors='ignore')

Lines=[]
for i in range(1000):
    line=file.readline()
    Lines.append(line)

import en_core_web_sm
nlp = en_core_web_sm.load(disable=['ner', 'parser'])

def cleaning(doc):
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = [token.lemma_ for token in doc if not token.is_stop]
    # Word2Vec uses context words to learn the vector representation of a target word,
    # if a sentence is only one or two words long,
    # the benefit for the training is very small
    if len(txt) > 2:
        return ' '.join(txt)
    
brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in Lines)

t = time()

txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=-1)]

print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))

df_clean = pd.DataFrame({'clean': txt})
df_clean = df_clean.dropna().drop_duplicates()
from gensim.models.phrases import Phrases, Phraser
sent = [row.split() for row in df_clean['clean']]
phrases = Phrases(sent, min_count=30, progress_per=100)

bigram = Phraser(phrases)
sentences = bigram[sent]

word_freq = defaultdict(int)
for sent in sentences:
    for i in sent:
        word_freq[i] += 1
len(word_freq)

print("top10 frequent words in our dataset are:")
print(sorted(word_freq, key=word_freq.get, reverse=True)[:10])

#training the model
import multiprocessing

from gensim.models import Word2Vec
cores = multiprocessing.cpu_count() 
w2v_model = Word2Vec(min_count=20,
                     window=2,
                     size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores-1)
#build vocab table
t = time()

w2v_model.build_vocab(sentences, progress_per=100)

print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

#training the model
t = time()

w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
w2v_model.init_sims(replace=True)
#most similar to

print("most similar words to american are:")
print(w2v_model.wv.most_similar(positive=["american"]))
print("most similar words to pandemic are:")
print(w2v_model.wv.most_similar(positive=["pandemic"]))
print("most similar words to election are:")
print(w2v_model.wv.most_similar(positive=["election"]))
print("most similar words to trump are:")
print(w2v_model.wv.most_similar(positive=["trump"]))

print("most similarity between words trump and president:")
print(w2v_model.wv.similarity("trump", 'president'))
print("most similarity between words america and nixon:")
print(w2v_model.wv.similarity('america', 'nixon'))
print("the most disimilar of the words trump, donald, inida is:")
print(w2v_model.wv.doesnt_match(['trump', 'donald', 'india']))

import numpy as np
import matplotlib.pyplot as plt

 
import seaborn as sns
sns.set_style("darkgrid")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def tsnescatterplot(model, word, list_names):
    """ Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,
    its list of most similar words, and a list of words.
    """
    arrays = np.empty((0, 300), dtype='f')
    word_labels = [word]
    color_list  = ['red']

    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)
    
    # gets list of most similar words
    close_words = model.wv.most_similar([word])
    
    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)
    # adds the vector for each of the words from list_names to the array
    for wrd in list_names:
        wrd_vector = model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)
        
    # Reduces the dimensionality from 300 to 50 dimensions with PCA
    reduc = PCA(n_components=19).fit_transform(arrays) 
    
    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)
    
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)
    
    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})
    
    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)
    
    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )
    
    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + df["words"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(15)

    
    plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
    plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
            
    plt.title('t-SNE visualization for {}'.format(word.title()))


tsnescatterplot(w2v_model, 'trump', ['businessman', 'actor', 'screenwriter', 'korea', 'india', 'canadian', 'tower', 'lunch'])
tsnescatterplot(w2v_model, 'hollywood', [i[0] for i in w2v_model.wv.most_similar(negative=["trump"])])
tsnescatterplot(w2v_model, "pandemic", [t[0] for t in w2v_model.wv.most_similar(positive=["pandemic"], topn=20)][10:])
print ("The vocabulary of our wiki documents is {}".format(len(w2v_model.wv.vocab)))

X=w2v_model[w2v_model.wv.vocab]

from nltk.cluster import KMeansClusterer
import nltk
import numpy as np 
  
from sklearn import cluster
from sklearn import metrics 

t=time() 
from sklearn.cluster import KMeans
#kmeans for 10 values of k
kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X)
                    for k in range(1, 10)]

# reduc = PCA(n_components=19).fit_transform(arrays)

def Kmeans_Inertia_Score(XD,title,kmeans_per_k):
    
    inertias = [model.inertia_ for model in kmeans_per_k]

    plt.figure(figsize=(8, 3.5))
    plt.plot(range(1, 10), inertias, "bo-")
    plt.xlabel("$k$", fontsize=14)
    plt.ylabel("Inertia", fontsize=14)
    plt.annotate('Elbow',
                 xy=(5, inertias[3]),
                 xytext=(0.55, 0.55),
                 textcoords='figure fraction',
                 fontsize=16,
                 arrowprops=dict(facecolor='black', shrink=0.1)
                )
    plt.title("inertia score against k "+title)
    plt.show()
#inertia score
print("KMEANS Inertia- Mean square error score")
Kmeans_Inertia_Score(X,'',kmeans_per_k)

#Kmeans_Inertia_Score(X2D,' for nPCA=2',kmeans_per_k_2DPCA)
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

t=time()    
kmeans = cluster.KMeans(n_clusters=5)
kmeans.fit(X)
  
labels = kmeans.labels_
centroids = kmeans.cluster_centers_ 

print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
print ("Cluster id labels for inputted data")
print (labels)
print ("Centroids data")
print (centroids)
 
print ("Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
print (kmeans.score(X))
 
silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
 
print ("Silhouette_score: ")
print (silhouette_score)


























