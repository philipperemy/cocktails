# coding: utf-8

# # LSTM IMDB Movie Review Tutorial
# Josiah Olson

# In[2]:

from __future__ import print_function

import time

import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge, BatchNormalization

from keras.preprocessing.text import Tokenizer

# In[3]:

max_features = 10000

X = []
Y = []

X.extend([open('../data/pos.txt.raw').readlines()][0])
pos_len = len(X)
Y.extend([1 for _ in range(pos_len)])

X.extend([open('../data/neg.txt.raw').readlines()][0])
Y.extend([0 for _ in range(len(X) - pos_len)])


def shuffle_in_unison_inplace(list1, list2):
    assert len(list1) == len(list2)
    from random import shuffle
    # Given list1 and list2
    o_list1 = []
    o_list2 = []
    shuffle_indices = range(len(list1))
    shuffle(shuffle_indices)
    for i in shuffle_indices:
        o_list1.append(list1[i])
        o_list2.append(list2[i])
    return o_list1, o_list2


X, Y = shuffle_in_unison_inplace(X, Y)

cutoff = int(len(X) * 0.75)

X_train = X[:cutoff]
y_train = Y[:cutoff]

X_test = X[cutoff:]
y_test = Y[cutoff:]

# In[6]:

# tokenize works to list of integers where each integer is a key to a word
imdbTokenizer = Tokenizer(nb_words=max_features)
imdbTokenizer.fit_on_texts(X_train)

# In[7]:

# print top 20 words
# note zero is reserved for non frequent words
for word, value in imdbTokenizer.word_index.items():
    if value < 20:
        print(value, word)

# In[8]:

# create int to word dictionary
intToWord = {}
for word, value in imdbTokenizer.word_index.items():
    intToWord[value] = word

# add a symbol for null placeholder
intToWord[0] = "!!!NA!!!"

print(intToWord[1])
print(intToWord[2])
print(intToWord[32])

# In[9]:

# convert word strings to integer sequence lists
print(X_train[0])
print(imdbTokenizer.texts_to_sequences(X_train[:1]))
for value in imdbTokenizer.texts_to_sequences(X_train[:1])[0]:
    print(intToWord[value])

X_train = imdbTokenizer.texts_to_sequences(X_train)
X_test = imdbTokenizer.texts_to_sequences(X_test)

# In[12]:

# Censor the data by having a max review length (in number of words)

# use this function to load data from keras pickle instead of munging as shown above
# (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features,
#                                                      test_split=0.2)

# cut texts after this number of words (among top max_features most common words)
max_len = max(max([len(v) for v in X_train]), max([len(v) for v in X_test]))

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=max_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

# In[13]:

# example of a sentence sequence, note that lower integers are words that occur more commonly
print("x:", X_train[0])  # per observation vector of 20000 words
print("y:", y_train[0])  # positive or negative review encoding

# In[14]:

# double check that word sequences behave/final dimensions are as expected
print("y distribution:", np.unique(y_train, return_counts=True))
print("max x word:", np.max(X_train), "; min x word", np.min(X_train))
print("y distribution test:", np.unique(y_test, return_counts=True))
print("max x word test:", np.max(X_test), "; min x word", np.min(X_test))

# as expected zero is the highly used word for words not in index

# set model hyper parameters
epochs = 100
embedding_neurons = 8
lstm_neurons = 16
batch_size = 4

USE_GOOGE = False

if USE_GOOGE:
    from gensim.models import Word2Vec

    googlew2v = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    # get word vectors for words in my index
    googleVecs = []
    for value in range(max_features):
        try:
            googleVecs.append(googlew2v[intToWord[value]])
        except:
            googleVecs.append(np.random.uniform(size=300))

    googleVecs = np.array(googleVecs)
    print(googleVecs)
    print(googleVecs.shape)

# this is the placeholder tensor for the input sequences
sequence = Input(shape=(max_len,), dtype='int32')
# this embedding layer will transform the sequences of integers
# into vectors of size embedding
# embedding layer converts dense int input to one-hot in real time to save memory

if USE_GOOGE:
    embedded = Embedding(max_features, 300, input_length=max_len, weights=[googleVecs])(sequence)
else:
    embedded = Embedding(max_features, embedding_neurons, input_length=max_len)(sequence)

bnorm = BatchNormalization()(embedded)

forwards = LSTM(lstm_neurons, dropout_W=0.4, dropout_U=0.4)(bnorm)
backwards = LSTM(lstm_neurons, dropout_W=0.4, dropout_U=0.4, go_backwards=True)(bnorm)

merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
after_dp = Dropout(0.5)(merged)
output = Dense(1, activation='sigmoid')(after_dp)

model_bidir_google = Model(input=sequence, output=output)

print(model_bidir_google.summary())

# In[171]:

# Bi-directional google

model_bidir_google.compile('rmsprop', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
start_time = time.time()

history_bidir_google = model_bidir_google.fit(X_train, y_train,
                                              batch_size=batch_size,
                                              nb_epoch=epochs,
                                              validation_data=[X_test, y_test])

end_time = time.time()
average_time_per_epoch = (end_time - start_time) / epochs
print("avg sec per epoch:", average_time_per_epoch)

# In[184]:
# test = "I would want to see this movie."
# test = imdbTokenizer.texts_to_sequences([test])
# from keras.preprocessing import sequence
# test = sequence.pad_sequences(test, maxlen=max_len)
# model_bidir_rmsprop.predict(test)
