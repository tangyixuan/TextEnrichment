'''
word2vec (follow https://www.tensorflow.org/tutorials/word2vec)
    (implemented here) skip-gram: predict context based on words, suitable for large data
    CBOW: predict words based on context, suitable for small data
NCE: noise contrastive estimation loss, tf.nn.nce_loss()
'''

import collections
import math
import os
import random
import zipfile
import numpy as np
from six.moves import urllib
import tensorflow as tf

# =========================== read in data ================================
print('==\nReading data...')
url = 'http://mattmahoney.net/dc/'

def verify_download(filename, expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified,',filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify '+filename)
    return filename
 
filename = verify_download('./data/text8.zip', 31344016)

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
#         print(len(f.read(f.namelist()[0]).split()))
#         print(f.read(f.namelist()[0]).split()[:50])
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words = read_data(filename) # array of words, without punctuation
# print(words[:20])
print('Data size:', len(words)) # datasize: 17 million words


# =========================== build vocabulary ================================
print('==\nBuilding vocabulary...')

vocab_size = 50000

def build_dataset(words):
    count = [['UNK',-1]] # [word, count]
    count.extend(collections.Counter(words).most_common(vocab_size-1))
    word_to_id = dict()
    for word, _ in count:
        word_to_id[word] = len(word_to_id)
    data = list()
    unk_count = 0
    for word in words:
        if word in word_to_id:
            index = word_to_id[word]
        else:
            unk_count+=1
            index = 0
        data.append(index)
    count[0][1]=unk_count
    id_to_word = dict(zip(word_to_id.values(),word_to_id.keys()))
    return data, count, word_to_id, id_to_word

data, count, word_to_id, id_to_word = build_dataset(words)
del words
print('Most common words (including UNK):', count[:10])
print('Sample data:', data[:20])
print('Corresponding words:', [id_to_word[i] for i in data[:20]])


# =========================== generate training batches ================================
print('==\nGenerating batches...')
data_index = 0

# a buffer and a global data_index are used here to generate new batches automatically
def generate_batch(batch_size, num_skips, skip_window):
    global data_index #it's the data_index defined globally, and it can be modified in this function
    assert batch_size % num_skips == 0 # num_skip is the number of samples generated for one center word, 
    assert num_skips <= 2* skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size,1), dtype=np.int32)
    
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span) # a queue that contains last #span elements that it sees, i.e. FIFO
    # here it's used to capture the context of a center word, it's the sliding window
    
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index+1)%len(data)
        
    for i in range(batch_size//num_skips):
        target = skip_window # index of the target word
        target_to_avoid = [ skip_window ] # can't sample the target word
        for j in range(num_skips):
            while target in target_to_avoid: # find a word that hasn'd been sampled before
                target = random.randint(0, span-1)
            target_to_avoid.append(target)
            batch[i*num_skips+j] = buffer[skip_window]
            labels[i*num_skips+j,0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index +1)%len(data)
        
    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)

print('Example for 1 batch (center word to context word):')
for i in range(8):
    print(batch[i],'(', id_to_word[batch[i]],') ->', labels[i,0],'(', id_to_word[labels[i,0]],')')      


# =========================== building network model ================================
print('==\nBuilding network model...')
# paras
batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2
num_iterations = 100001
save_step = 20000
print_step = 2000

valid_size = 16 # number of words for validation
valid_window = 100 # only sample validation words from the most frequent 100 words
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
valid_step = 10000
num_sampled = 64 # number of negative sampling

# model: placeholder, variables, layers
graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size,1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    
    with tf.device('/cpu:0'):
        embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        
        nce_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=1.0/math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocab_size]))

    # loss, optimizer - NCE loss with weights and bias
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, 
                                         biases=nce_biases, 
                                         labels=train_labels, 
                                         inputs=embed, 
                                         num_sampled=num_sampled, 
                                         num_classes=vocab_size))
    
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True)) # L2 normalization
    normalized_embeddings = embeddings/norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset) # embedding for validation words
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True) 
    # similarity between embeddings of validation words and embeddings of all the words
    # in order to find out the most similar words for validation words
    
# ================================ train ================================
print('==\nTraining...')
# saver = tf.train.Saver()
saver = tf.train.Saver([embeddings])

with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    
    average_loss = 0
    for step in range(num_iterations):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs:batch_inputs, train_labels: batch_labels}
        
        _, loss_val = sess.run([optimizer, loss], feed_dict = feed_dict)
        average_loss += loss_val
        
        if step % print_step == 0:
            if step > 0:
                average_loss/=print_step
            print ('average loss at step %d: %.4f'%(step, average_loss))
       
        # validation        
        if step % valid_step == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = id_to_word[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i,:]).argsort()[1:top_k+1]
                log_str = 'Nearest to %s:'%valid_word
                for k in range(top_k):
                    close_word = id_to_word[nearest[k]]
                    log_str = "%s %s,"%(log_str, close_word)
                print(log_str)

        # save to file            
        if step % save_step == 0:
            save_path = saver.save(sess, "./result/embedding_%d.ckpt"%step)
            print("Variables saved in file: %s" % save_path)

    final_embeddings = normalized_embeddings.eval()

