
# coding: utf-8

# Word2vec from Manaus
# =============
# 
# Originally from [Udacity Course](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/5_word2vec.ipynb) assignment 5.
# 
# What it does
# ------------
# 
# Accept list of lists of keyword and predict the similarity of keywords according to their companions.
#
# GPU or CPU?
# ----------
# 
# Unless you have NVidia, it's CPU because Tensorflow does not support OpenCL. At the moment it does not work with [tf-coriander](https://github.com/hughperkins/tf-coriander).
# 

from __future__ import print_function
from __future__ import division
import itertools
import collections
import math
import numpy as np
from scipy.spatial.distance import cosine
import os
import random
import tensorflow as tf
import re
import json
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve

filename = 'xyz.tsv'
# The format of our input file is, for each line:
# "id";"keyword1|0.25 keyword2|0.24 keyword3|0.0848 keyword4|0.14"
# i.e. after the ID which identifies the conversation, a list of keywords with their score.

def read_data(filename):
    """Extract all words all alpha and not uppercase
    words: list of all words
    sentences: list of lists, each list a sentence
    sentences_index: a list, where each element says how to find that word in sentences: (4,3) word[3] in sentence[4]
    """
    # list of list of words
    sentences = list()
    # all words in order of apparence (repeated)
    words = list()
    # (sentence_index_in_sentences, word_index_in_sentence) => index in words
    sentences_index_dict = dict()
    # index in words => (sentence_index_in_sentences, word_index_in_sentence)
    sentences_index = []
    with open(filename) as f:
        sentence_count = 0
        for line in f.readlines():
            sentence = list()
            word_count = 0
            for word in line.replace('"', ' ').replace('|', ' ').replace('";"', ' ').split():
                if word.isalpha() and not word.isupper():
                    #print(word)
                    words.append(word)
                    sentence.append(word)
                    sentences_index_dict[(sentence_count, word_count)] = len(words) - 1
                    sentences_index.append((sentence_count, word_count))
                    word_count += 1
            sentences.append(sentence)
            sentence_count += 1
                
    return words, sentences, sentences_index, sentences_index_dict
  
words, sentences, sentences_index, sentences_index_dict = read_data(filename)
print('Data size %d' % len(words))

print(words[:9])
print(sentences[:5])
print(sentences_index[:8])
print(sentences_index_dict[(0, 0)])


#### Add possible synonyms

# Libraries for Lev distance
def _edit_dist_init(len1, len2):
    lev = []
    for i in range(len1):
        lev.append([0] * len2)  # initialize 2D array to zero
    for i in range(len1):
        lev[i][0] = i           # column 0: 0,1,2,3,4,...
    for j in range(len2):
        lev[0][j] = j           # row 0: 0,1,2,3,4,...
    return lev


def _edit_dist_step(lev, i, j, s1, s2, substitution_cost=1, transpositions=False):
    c1 = s1[i - 1]
    c2 = s2[j - 1]

    # skipping a character in s1
    a = lev[i - 1][j] + 1
    # skipping a character in s2
    b = lev[i][j - 1] + 1
    # substitution
    c = lev[i - 1][j - 1] + (substitution_cost if c1 != c2 else 0)

    # transposition
    d = c + 1  # never picked by default
    if transpositions and i > 1 and j > 1:
        if s1[i - 2] == c2 and s2[j - 2] == c1:
            d = lev[i - 2][j - 2] + 1

    # pick the cheapest
    lev[i][j] = min(a, b, c, d)


def edit_distance(s1, s2, substitution_cost=1, transpositions=False):
    """
    Calculate the Levenshtein edit-distance between two strings.
    The edit distance is the number of characters that need to be
    substituted, inserted, or deleted, to transform s1 into s2.  For
    example, transforming "rain" to "shine" requires three steps,
    consisting of two substitutions and one insertion:
    "rain" -> "sain" -> "shin" -> "shine".  These operations could have
    been done in other orders, but at least three steps are needed.

    Allows specifying the cost of substitution edits (e.g., "a" -> "b"),
    because sometimes it makes sense to assign greater penalties to substitutions.

    This also optionally allows transposition edits (e.g., "ab" -> "ba"),
    though this is disabled by default.

    :param s1, s2: The strings to be analysed
    :param transpositions: Whether to allow transposition edits
    :type s1: str
    :type s2: str
    :type substitution_cost: int
    :type transpositions: bool
    :rtype int
    """
    # set up a 2-D array
    len1 = len(s1)
    len2 = len(s2)
    lev = _edit_dist_init(len1 + 1, len2 + 1)

    # iterate over the array
    for i in range(len1):
        for j in range(len2):
            _edit_dist_step(lev, i + 1, j + 1, s1, s2,
                            substitution_cost=substitution_cost, transpositions=transpositions)
    return lev[len1][len2]

def jaccard_distance(label1, label2):
    """Distance metric comparing set-similarity.

    """
    return (len(label1.union(label2)) - len(label1.intersection(label2)))/len(label1.union(label2))

def masi_distance(label1, label2):
    """Distance metric that takes into account partial agreement when multiple
    labels are assigned.

    >>> from nltk.metrics import masi_distance
    >>> masi_distance(set([1, 2]), set([1, 2, 3, 4]))
    0.335

    Passonneau 2006, Measuring Agreement on Set-Valued Items (MASI)
    for Semantic and Pragmatic Annotation.
    """

    len_intersection = len(label1.intersection(label2))
    len_union = len(label1.union(label2))
    len_label1 = len(label1)
    len_label2 = len(label2)
    if len_label1 == len_label2 and len_label1 == len_intersection:
        m = 1
    elif len_intersection == min(len_label1, len_label2):
        m = 0.67
    elif len_intersection > 0:
        m = 0.33
    else:
        m = 0

    return (1 - (len_intersection / float(len_union))) * m



def interval_distance(label1,label2):
    """Krippendorff's interval distance metric

    >>> from nltk.metrics import interval_distance
    >>> interval_distance(1,10)
    81

    Krippendorff 1980, Content Analysis: An Introduction to its Methodology
    """

    try:
        return pow(label1 - label2, 2)
#        return pow(list(label1)[0]-list(label2)[0],2)
    except:
        print("non-numeric labels not supported with interval distance")



def presence(label):
    """Higher-order function to test presence of a given label
    """

    return lambda x, y: 1.0 * ((label in x) == (label in y))



def fractional_presence(label):
    return lambda x, y:        abs(((1.0 / len(x)) - (1.0 / len(y)))) * (label in x and label in y)         or 0.0 * (label not in x and label not in y)         or abs((1.0 / len(x))) * (label in x and label not in y)         or ((1.0 / len(y))) * (label not in x and label in y)



def custom_distance(file):
    data = {}
    with open(file, 'r') as infile:
        for l in infile:
            labelA, labelB, dist = l.strip().split("\t")
            labelA = frozenset([labelA])
            labelB = frozenset([labelB])
            data[frozenset([labelA,labelB])] = float(dist)
    return lambda x,y:data[frozenset([x,y])]

def binary_distance(label1, label2):
    """Simple equality test.

    0.0 if the labels are identical, 1.0 if they are different.

    >>> from nltk.metrics import binary_distance
    >>> binary_distance(1,1)
    0.0

    >>> binary_distance(1,3)
    1.0
    """

    return 0.0 if label1 == label2 else 1.0

def syn_distance(w1, w2, ngram=3):
    steps = max(len(w1), len(w2))
#    print('steps: ', steps)
    d = 0.0
    for s in range(steps-ngram):
#        print('prima', s, d)
#        print('distance for:', w1[s:s+ngram], w2[s:s+ngram], ,': ', edit_distance(w1[s:s+ngram], w2[s:s+ngram],  transpositions=True) )
        d += edit_distance(w1[s:s+ngram], w2[s:s+ngram],  transpositions=True) * math.exp(-s)
#        print('dopo', s, d)
    return d


def demo():
    edit_distance_examples = [
        ("install", "installation"), ("abcdef", "acbdef"), ("implementation", "installation"),
        ("implementation", "implement"),
        ("language", "lnaugage"), ("licence", "license"), ('ronny', 'ronnie'), ('interestingly', 'interrupted')]
    for s1, s2 in edit_distance_examples:
        print("Syn distance between '%s' and '%s':" % (s1, s2), syn_distance(s1, s2, 4))
#     for s1, s2 in edit_distance_examples:
#         print("Edit distance with transpositions between '%s' and '%s' with Transpo:" % (s1, s2), edit_distance(s1, s2, transpositions=True))
#     for s1, s2 in edit_distance_examples:
#         print("Edit distance with transpositions between '%s' and '%s' NO Transpo:" % (s1, s2), edit_distance(s1, s2, transpositions=False))
#     for s1, s2 in edit_distance_examples:
#         print("TRUNCATED Edit distance with transpositions between '%s' and '%s':" % (s1, s2), edit_distance(s1[4:min(8, len(s1), len(s2))], s2[4:min(8, len(s1), len(s2))]))
#     for s1, s2 in edit_distance_examples:
#         print("Jaccard distance between '%s' and '%s':" % (s1, s2), jaccard_distance(set(s1), set(s2)))
#     for s1, s2 in edit_distance_examples:
#         print("MASI distance between '%s' and '%s':" % (s1, s2), masi_distance(set(s1), set(s2)))


demo()

def syn_distance(w1, w2, ngram=3):
    steps = max(len(w1), len(w2))
    print('steps: ', steps)
    d = 0.0
    for s in range(steps-ngram):
        print('prima', s, d)
        print('distance for:', w1[s:s+ngram], w2[s:s+ngram], ': ', edit_distance(w1[s:s+ngram], w2[s:s+ngram],  transpositions=True) )
        d += edit_distance(w1[s:s+ngram], w2[s:s+ngram],  transpositions=True) * math.exp(-s)
        print('dopo', s, d)
    return d
syn_distance('instructions', 'installtion', 4)


def synonyms_candidates(words, cut=0.1, ngram=3):
    words = set(words)
    syn_sentences = []
    while len(words) > 1:
        w = words.pop()
        sentence = [w]
        for w2 in words:
            #L = min(8, len(w), len(w2))
            #if w[:4] == w2[:4] and edit_distance(w[4:L], w2[4:L]) < 2:
            if syn_distance(w, w2, ngram) < cut and min(len(w), len(w2)) > ngram:
                sentence.append(w2)
        words.difference_update(set(sentence))
        if len(sentence) > 1:
            syn_sentences.append(sentence)
    return(syn_sentences)


try:
    with open("synonyms.json", 'r') as f:
        synonyms = json.load(f)
    print("Read from file ", len(synonyms), " synonyms: ", synonyms[:10])
        
except:
    print("Producing synonyms list")
    synonyms = ['pippo', 'pipo']  # synonyms_candidates(set(words))
    with open("synonyms.json", 'w') as f:
        json.dump(synonyms, f)


#### Build the dictionary variables

def build_dataset(sentences, words):
    """
    Returns:
    data: sentences, but with each word is substituted by its ranking in dictionary
    count: list of (word, occurrence)
    dictionary: dict, word -> ranking
    reverse_dictionary: dict, ranking -> word
    """
    count = list()
    #NB We are using keywords, therefore no filters!
    count.extend(collections.Counter(words).most_common())
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = [[dictionary[w] for w in sentence] for sentence in sentences if len(sentences) > 0]
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
    return data, count, dictionary, reverse_dictionary

# data, count, dictionary, reverse_dictionary = build_dataset(sentences, words)
# try with synonyms:
synonym_data, count, dictionary, reverse_dictionary = build_dataset(synonyms, words)
sentence_data, count, dictionary, reverse_dictionary = build_dataset(sentences, words)
vocabulary_size = len(count)
print('Most common words:', count[:10])  # count is never used
print('Sample synonym data', synonym_data[:10])
print('Sample sentence data', sentence_data[:10])

assert(dictionary.get('the') is None)  # If there is 'the', you havent used a good statistical extractor


#### Generate a training batch for the skip-gram model.

from random import shuffle

data_index = 0

def generate_batch(data, data_index):
    '''
    IN
    data: XXX
    data_index: index of sentence
    OUT
    batch: nparray (variable length) of words
    label: nparray (same length as batch, 1) of context words
    data_index: data_index + 1
    '''
    if len(data[data_index]) < 2:
        return None, None, (data_index + 1) % len(data)
    else:
        combinations = np.asarray([w for w in itertools.product(data[data_index][:12], data[data_index][:12]) if w[0] != w[1]], dtype=np.int32)

        batch, l = combinations.T
        #labels = np.asarray([l], dtype=np.int32)
        labels = np.asarray(l, dtype=np.int32)
        del(l)
        return batch, labels, (data_index + 1) % len(data)



# print('data:', [reverse_dictionary[di] for di in data[:8]])

# for num_skips, skip_window in [(2, 1), (4, 2)]:
#     data_index = 0
#     batch, labels = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
#     print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
#     print('    batch:', [reverse_dictionary[bi] for bi in batch])
#     print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])

def distanza(x, y):
    return np.dot(x / np.linalg.norm(x), y / np.linalg.norm(y))

sess = tf.InteractiveSession()
c = tf.constant([[4.0], [5.0], [6.0]])
print(c.eval())
d = tf.reshape(c, [3])
print(d.eval())
sess.close()

#batch, labels, data_index = generate_batch(data, 0)
#train_labels = tf.convert_to_tensor(labels, dtype=tf.int32)
#train_batch = tf.convert_to_tensor(batch, dtype=tf.int32)

#embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
#embedded_input = tf.nn.embedding_lookup(embeddings, train_batch)
#embedded_labels = tf.nn.embedding_lookup(embeddings, train_labels)

#distances_matrix = embedded_labels @ tf.transpose(embeddings)
#distances_matrix = tf.matmul(embedded_labels, tf.transpose(embeddings))


### Embedding size

embedding_size = 64


## First graph: SYNONYMS

graph = tf.Graph()
with graph.as_default(), tf.device('/cpu:0'):

    # Input data.
    train_batch = tf.placeholder(tf.int32, shape=[None])
    train_labels = tf.placeholder(tf.int32, shape=[None])

    ## Random values to the embedding vectors: M(vocabulary x embedding size)
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    
    embedded_inputs = tf.nn.embedding_lookup(embeddings, train_batch)
    embedded_labels = tf.nn.embedding_lookup(embeddings, train_labels)

    # matrix of embeddings.T * embedded_inputs, i.e. making the
    # scalar product of each embedded word (embedding.T is |V| x d)
    # with the input. This is a rough measurement of the distance,
    # which must be small for the labels
    distances_matrix = tf.matmul(embedded_inputs, tf.transpose(embeddings))

    one_hot_labels = tf.one_hot(train_labels, depth=vocabulary_size)
    xe = tf.losses.softmax_cross_entropy(one_hot_labels, distances_matrix)

    # The optimizer will optimize the softmax_weights AND the embeddings.
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(xe)

num_steps = 2001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    average_loss = 0.0
    data_index = 0
    for step in range(num_steps):
        try:
#            if step % int(num_steps / 10) == 0:
#                print("Distanza prima: ", distanza(1395, 810))
            batch_data, batch_labels, data_index = generate_batch(synonym_data, data_index)
            feed_dict = {train_batch : batch_data, train_labels : batch_labels}
            _, lozz = session.run([optimizer, xe], feed_dict=feed_dict)
            average_loss += lozz

            if step % int(num_steps / 10) == 0:
                print("Done step ", step)
                average_loss = average_loss / float(num_steps / 10)
                print("Average loss:", average_loss)    
                average_loss = 0.0
                embeds = embeddings.eval()
                
                print("Distanza: ", distanza(embeds[1395], embeds[810]))            
        except:
            print("Problems with data_index = ", data_index)
            data_index += 1
            
      
    distance_embeddings = embeddings.eval()

distanza(distance_embeddings[dictionary['companies']], distance_embeddings[dictionary['company']])


## Actual word2vec 
### (To be done CBOW)

num_sampled = 64 # Number of negative examples to sample.

# Random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. 
valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution.
#valid_examples = np.array(random.sample(range(valid_window), valid_size))
valid_examples = np.array([67 , 53 , 73 , 26 , 30 , 65 , 15 , 41 , 55 , 40 , 7 , 31 , 98 , 48 , 36 , 88])

with graph.as_default(), tf.device('/cpu:0'):
 
    ## Uncomment only for testing w/o synonyms
    # embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

    train_labels_one = tf.placeholder(tf.int32, shape=[None, 1])
    
    # Words to test on
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    ## Weights and biases
    softmax_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                         stddev=1.0 / math.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the softmax loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(weights=softmax_weights, 
                                   biases=softmax_biases, 
                                   inputs=embedded_inputs,
                                   labels=train_labels_one, 
                                   num_sampled=num_sampled, 
                                   num_classes=vocabulary_size))
    
    # Optimizer.
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

    # Compute the similarity between minibatch examples and all embeddings.
    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

valid_examples

num_steps = 500001
data_index = 0

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized/2')
    average_loss = 0
    for step in range(num_steps):
        batch_data, batch_labels, data_index = generate_batch(sentence_data, data_index)
        if batch_data is None:
            #print("No batch_data")
            continue
        # TODO This is horrible I guess
        batch_labels_one = batch_labels.reshape([batch_labels.shape[0], 1])
        feed_dict = {train_batch : batch_data, train_labels_one: batch_labels_one}
        _, lozz = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += lozz
        if step % int(num_steps / 10) == 0:
            print("Done step ", step)
            average_loss = average_loss / float(num_steps / 10)
            print("Average loss:", average_loss)    
            average_loss = 0.0

            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8 # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log = 'Nearest to %s:' % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)
    final_embeddings = normalized_embeddings.eval()



valid_word = 'google'
i = dictionary[valid_word]
top_k = 8 # number of nearest neighbors
nearest = (-sim[i, :]).argsort()[1:top_k+1]
log = 'Nearest to %s:' % valid_word
for k in xrange(top_k):
    close_word = reverse_dictionary[nearest[k]]
    log = '%s %s,' % (log, close_word)
print(log)

print(distanza(1395, 810))
print(distanza(2, 2))
np.linalg.norm(final_embeddings[9])