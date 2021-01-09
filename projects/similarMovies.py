#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This file is a small project that returns clusters and plots a
dendogram of a small subset of movies from IMDb. It mainly uses the
Natural Language Toolkit and TfidfVectorizer Data from the homonimous
Datacamp project.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

import nltk
from nltk.stem.snowball import SnowballStemmer # The stemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram


# function performing both stemming and tokenization
def tokenize_and_stem(text):
    
    # Tokenize by sentence, then by word
    tokens = []
    for sent in nltk.sent_tokenize(text):
        tokens.extend(word for word in nltk.word_tokenize(sent))
    
    # Filter out raw tokens to remove noise
    filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]
    
    # Stem the filtered_tokens
    stemmer = SnowballStemmer("english")
    stems = [stemmer.stem(word) for word in filtered_tokens]
    return stems


def kmeansClusters(movies_df, tfidf_matrix, nCl=5):
    km = KMeans(n_clusters=nCl)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()
    movies_df["cluster"] = clusters
    for i in range(nCl):
        print('\n')
        print('Cluster {}: {}'.format(i,movies_df[movies_df['cluster']==i].index.to_list()))
       
    return movies_df
    
    
def dendogramPlot(movies_df, tfidf_matrix):
   
    similarity_distance = 1 - cosine_similarity(tfidf_matrix)
    # Create mergings matrix 
    mergings = linkage(similarity_distance, method='complete')
    
    # Plot the dendrogram, using title as label column
    dendrogram_ = dendrogram(mergings,
                   labels=[x for x in movies_df.index],
                   leaf_rotation=90,
                   leaf_font_size=13,
    )
    # Adjust the plot
    fig = plt.gcf()
    _ = [lbl.set_color('r') for lbl in plt.gca().get_xmajorticklabels()]
    fig.set_size_inches(160, 30)
    # Show the plotted dendrogram
    plt.show()
    

    
if __name__ == "__main__":
    filename = input('Insert the name of the movies database or \
ENTER for the default value "movies.csv": ')
    if filename == '': 
        filename = 'datasets/movies.csv'
    
    movies_df = pd.read_csv(filename, index_col='title')
    movies_df['plot'] = movies_df['wiki_plot'].astype(str) + "\n" + \
                 movies_df['imdb_plot'].astype(str)
    # Print important features
    print('# Movies: {}\n'.format(len(movies_df)))
    print('Columns: {}\n'.format(list(movies_df.columns)))
    # Creatint the tfid matrix
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem,
                                 ngram_range=(1,3))
    tfidf_matrix = tfidf_vectorizer.fit_transform([x for x in movies_df["plot"]])
    kmeansClusters(movies_df, tfidf_matrix, 7)
    dendogramPlot(movies_df, tfidf_matrix)







