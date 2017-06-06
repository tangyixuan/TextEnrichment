import collections
import math
import os
import random
import zipfile
import numpy as np
from six.moves import urllib
import tensorflow as tf
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time

# select words after pca not before pca

def plot_with_labels(low_dim_embedings, labels, filename):
    assert low_dim_embedings.shape[0] >= len(labels), "More labels than embedding"
    plt.figure(figsize = (18,18))
    for i,label in enumerate(labels):
        x,y = low_dim_embedings[i,:]
        plt.scatter(x,y,c=color[i])
        plt.annotate(label,xy=(x,y),xytext=(5,2),textcoords='offset points',ha='right',va='bottom')    
    plt.savefig(filename)

# labels and low_dim_embeddings has the same index
# selected and colors has the same index
def plot_selected_with_labels_and_colors(low_dim_embedings, labels, selected, colors, filename):
    assert low_dim_embedings.shape[0] >= len(labels), "More labels than embedding"
    plt.figure(figsize = (18,18))
    for i in range(len(selected)):
        curr_label = selected[i]
        curr_color = colors[i]
        if curr_label in labels:
            j = np.where(labels==curr_label)[0][0]
            x,y = low_dim_embedings[j,:]
            plt.scatter(x,y,c=curr_color)
#             plt.scatter(x,y)
            plt.annotate(curr_label,xy=(x,y),xytext=(5,2),textcoords='offset points',ha='right',va='bottom')    
    plt.savefig(filename)

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

'''
####################### self trained embeddings on text8.zip #######################

id_to_word = np.load('./result/id2word.npy').item()

# self trained embedding
vocab_size = 50000
embedding_size = 128
embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True)) # L2 normalization
normalized_embeddings = embeddings/norm

saver = tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver.restore(sess, "./result/embedding_100000.ckpt")
    print("Variables restored.")
    final_embeddings = normalized_embeddings.eval()     

plot_only = 500
low_dim_embs =  tsne.fit_transform(final_embeddings[:plot_only,:])
labels = [id_to_word[i] for i in range(plot_only)]
# np.save('./result/words_top500.npy',labels)
plot_with_labels(low_dim_embs, labels, './image/text8.png')

'''


# labels = np.load('./result/words_top500.npy')
labels = np.load('./result/words_top2k.npy') 
# selected = np.load('./result/dm_keywords.npy')
selected = np.load('./result/testword.npy')
colors = np.load('./result/color.npy')

####################### Google word2vec #######################
'''
import gensim
print('==\nloading word2vec model...')
start_time = time.time()
w2v_model = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin.gz', binary=True)  
print('word2vec model loaded in %s seconds'%(time.time()-start_time))
w2v_embs = []
for w in labels:
    if w in w2v_model.vocab:
        w2v_embs.append(w2v_model[w])
    else:
        print(w)
        w2v_embs.append([0]*300)

w2v_low_embs =  tsne.fit_transform(w2v_embs)
# plot_with_labels(w2v_low_embs, labels, './image/w2v_color_2.png')
plot_selected_with_labels_and_colors(w2v_low_embs, labels, selected, colors,'./image/w2v_color_2k.png')
'''

####################### Glove common crawler #######################

def loadGloveModel(path):
    print("loading Glove model")
    start_time = time.time()
    f = open(path,'r')
    model = {}
    for line in f:
        parts = line.split()
        word = parts[0]
        embedding = [float(val) for val in parts[1:]]
        model[word] = embedding
    print("glove model: %d words loaded in %s seconds"%(len(model), (time.time() - start_time))) # 105s, around 2 mins
    return model

glove_embs = []
glove_model = loadGloveModel('./data/glove.42B.300d.txt')
for w in labels:
    if w in glove_model:
        glove_embs.append(glove_model[w])
    else:
        print(w)
        glove_embs.append([0]*300)

glove_low_embs =  tsne.fit_transform(glove_embs)
# plot_with_labels(glove_low_embs, labels, './image/glove_color_2.png') # 189s, around 3 mins
plot_selected_with_labels_and_colors(glove_low_embs, labels, selected, colors,'./image/glove_color_2k.png')
